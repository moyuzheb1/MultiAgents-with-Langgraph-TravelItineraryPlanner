from data.shenzhen import HOTELS, POIS, RESTAURANTS, all_entities


def test_seed_scale():
    assert len(POIS) >= 100
    assert len(RESTAURANTS) >= 70
    assert len(HOTELS) >= 10
    assert len(all_entities()) >= 190


def test_entity_ids_unique():
    ids = [e.id for e in all_entities()]
    assert len(ids) == len(set(ids))


def test_every_entity_has_area():
    for e in all_entities():
        assert getattr(e, "area", "")
