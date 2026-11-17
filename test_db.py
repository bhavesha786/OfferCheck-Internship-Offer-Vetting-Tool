import pytest

import db


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "test.db"
    db.init_db(p)
    return p


def _report(score=50, verdict="Proceed with caution"):
    return {"company": "Acme", "role": "Intern", "final_risk_score": score, "verdict": verdict, "red_flags": []}


def test_save_list_get_roundtrip(path):
    offer_id = db.save_offer(_report(), "offer text here", path)
    rows = db.list_offers(path)
    assert len(rows) == 1 and rows[0]["id"] == offer_id and rows[0]["status"] == "Considering"
    stored = db.get_offer(offer_id, path)
    assert stored["offer_text"] == "offer text here" and stored["report"]["company"] == "Acme"


def test_update_and_delete(path):
    offer_id = db.save_offer(_report(), "text", path)
    db.update_offer(offer_id, "Accepted", "joined!", path)
    row = db.list_offers(path)[0]
    assert row["status"] == "Accepted" and row["notes"] == "joined!"
    db.delete_offer(offer_id, path)
    assert db.list_offers(path) == []


def test_invalid_status_rejected(path):
    offer_id = db.save_offer(_report(), "text", path)
    with pytest.raises(ValueError):
        db.update_offer(offer_id, "Maybe", "", path)


def test_sql_injection_is_inert(path):
    nasty = "x'); DROP TABLE offers; --"
    db.save_offer({**_report(), "company": nasty}, nasty, path)
    assert db.list_offers(path)[0]["company"] == nasty
