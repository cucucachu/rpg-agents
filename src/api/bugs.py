"""Bug report endpoints for user-submitted issues linked to agent traces."""

import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from bson import ObjectId

from ..db import get_db
from ..models import BugReport, User
from .auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/worlds", tags=["bugs"])


class ReportBugRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Description of the issue")


class BugReportResponse(BaseModel):
    id: str
    trace_id: str
    user_id: str
    description: str


@router.post(
    "/{world_id}/messages/{message_id}/bugs",
    response_model=BugReportResponse,
    status_code=201,
)
async def report_bug(
    world_id: str,
    message_id: str,
    request: ReportBugRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Submit a bug report linked to a GM message's agent trace.

    Looks up the GM message to find its trace_id, then creates a BugReport
    document and appends its ID to the trace's bug_report_ids list.
    """
    db = await get_db()

    # Verify world access
    access = await db.world_access.find_one({
        "user_id": current_user.id,
        "world_id": world_id,
    })
    if not access:
        raise HTTPException(status_code=403, detail="No access to this world")

    # Look up the GM message
    try:
        msg_doc = await db.messages.find_one({"_id": ObjectId(message_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid message ID")

    if not msg_doc:
        raise HTTPException(status_code=404, detail="Message not found")

    if msg_doc.get("world_id") != world_id:
        raise HTTPException(status_code=403, detail="Message does not belong to this world")

    if msg_doc.get("message_type") != "gm":
        raise HTTPException(status_code=400, detail="Bug reports can only be filed against GM messages")

    trace_id = msg_doc.get("trace_id")
    if not trace_id:
        raise HTTPException(status_code=404, detail="No trace found for this message")

    # Create the bug report
    bug_report = BugReport(
        trace_id=trace_id,
        user_id=current_user.id,
        description=request.description,
    )
    result = await db.bug_reports.insert_one(bug_report.to_doc())
    bug_report_id = str(result.inserted_id)
    bug_report.id = bug_report_id

    # Link the bug report back to the trace
    await db.traces.update_one(
        {"_id": ObjectId(trace_id)},
        {"$push": {"bug_report_ids": ObjectId(bug_report_id)}},
    )

    logger.info(f"Bug report {bug_report_id} filed against trace {trace_id} by user {current_user.id}")

    return BugReportResponse(
        id=bug_report_id,
        trace_id=trace_id,
        user_id=current_user.id,
        description=request.description,
    )
