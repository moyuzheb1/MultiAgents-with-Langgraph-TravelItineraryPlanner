import pytest

from retrieval.graph_store import CypherError, parse_cypher

BAD = [
    ("bad_attr", "MATCH (n:POI) WHERE n.has_elevator = true RETURN n LIMIT 5"),
    ("bad_label", "MATCH (n:Scenic) RETURN n LIMIT 5"),
    ("bad_rel", "MATCH (n:POI)-[:FRIEND_OF]->(:Area) RETURN n LIMIT 5"),
    ("write", "CREATE (n:POI {name:'x'}) RETURN n"),
    ("multi_match", "MATCH (n:POI) MATCH (m:Restaurant) RETURN n LIMIT 5"),
]


@pytest.mark.parametrize("name,cypher", BAD)
def test_invalid_cypher_rejected(name, cypher):
    with pytest.raises(CypherError):
        parse_cypher(cypher)


def test_valid_cypher_accepted():
    q = parse_cypher(
        "MATCH (n:POI)-[:LOCATED_IN]->(:Area {name:'南山区'}) "
        "WHERE n.wheelchair = true RETURN n LIMIT 5"
    )
    assert q.label == "POI"
    assert q.area == "南山区"
