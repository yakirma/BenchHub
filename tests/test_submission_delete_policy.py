"""Authorization policy for archiving vs. permanently deleting submissions.

Delete is stricter than manage: on a PUBLIC board only an admin or the
submission's own owner may permanently delete — the board owner can archive
(hide) but not erase other people's work. NULL-owner (legacy/mirrored) rows are
never manageable by an arbitrary logged-in user. Regression guard for the fix
that closed the old `owner is None -> anyone` hole.
"""
from types import SimpleNamespace

from app import (
    _can_manage_submission, _can_delete_submission,
    db, User, Leaderboard, Dataset, Submission,
)


def _U(uid, admin=False):
    return SimpleNamespace(id=uid, is_admin=admin, email=f"u{uid}@ex.com")


def _S(owner, lb_owner, vis):
    return SimpleNamespace(owner_user_id=owner,
                           leaderboard=SimpleNamespace(owner_user_id=lb_owner, visibility=vis))


def test_public_board_delete_admin_and_sub_owner_only():
    s = _S(owner=1, lb_owner=3, vis="public")
    assert _can_delete_submission(_U(1), s)             # submission owner
    assert _can_delete_submission(_U(9, admin=True), s)  # admin
    assert not _can_delete_submission(_U(3), s)          # LB owner: NO on public
    assert not _can_delete_submission(_U(2), s)          # bystander: no
    assert _can_manage_submission(_U(3), s)              # LB owner may still archive


def test_nonpublic_board_owner_may_delete():
    for vis in ("unlisted", "private", None):
        assert _can_delete_submission(_U(3), _S(owner=1, lb_owner=3, vis=vis)), vis


def test_null_owner_not_manageable_by_bystander():
    s = _S(owner=None, lb_owner=3, vis="public")
    assert not _can_manage_submission(_U(2), s)
    assert not _can_delete_submission(_U(2), s)
    assert _can_manage_submission(_U(3), s)              # LB owner manages orphan row
    assert not _can_delete_submission(_U(3), s)          # but not delete on public


def test_delete_route_forbidden_for_non_owner(client, db_session):
    owner = User(email="own@ex.com", display_name="Owner",
                 oauth_provider="github", oauth_sub="own")
    other = User(email="oth@ex.com", display_name="Other",
                 oauth_provider="github", oauth_sub="oth")
    db.session.add_all([owner, other])
    db.session.flush()
    ds = Dataset(name="pol_ds")
    db.session.add(ds)
    db.session.flush()
    lb = Leaderboard(name="pol_lb", summary_metrics="",
                     visibility="public", owner_user_id=owner.id)
    lb.datasets.append(ds)
    db.session.add(lb)
    db.session.flush()
    sub = Submission(name="m", leaderboard_id=lb.id,
                     processing_status="Processed", owner_user_id=owner.id)
    db.session.add(sub)
    db.session.commit()
    sub_id = sub.id

    with client.session_transaction() as sess:
        sess["user_id"] = other.id
    resp = client.post(f"/delete_submission/{sub_id}")
    assert resp.status_code == 403
    assert Submission.query.get(sub_id) is not None   # survived the forbidden delete
