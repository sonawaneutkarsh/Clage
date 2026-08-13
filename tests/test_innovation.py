import pytest

from neat.innovation import DEFAULT_NEXT_INNOVATION, DEFAULT_NEXT_NODE_ID, InnovationDB


def test_defaults_start_after_fixed_interface():
    db = InnovationDB()
    assert db.new_node_id() == DEFAULT_NEXT_NODE_ID
    assert db.connection_innovation(0, 10) == DEFAULT_NEXT_INNOVATION


def test_connection_innovation_mints_sequentially():
    db = InnovationDB()
    assert db.connection_innovation(0, 10) == 1
    assert db.connection_innovation(1, 11) == 2
    assert db.connection_innovation(2, 12) == 3


def test_connection_innovation_reuses_same_pair():
    db = InnovationDB()
    first = db.connection_innovation(0, 10)
    again = db.connection_innovation(0, 10)
    assert first == again == 1


def test_connection_innovation_different_pairs_differ():
    db = InnovationDB()
    a = db.connection_innovation(0, 10)
    b = db.connection_innovation(0, 11)
    assert a != b


def test_add_node_innovation_mints_node_id_and_two_innovations():
    db = InnovationDB()
    ni = db.add_node_innovation(0, 10)
    assert ni.node_id == 14
    assert ni.in_innovation == 1
    assert ni.out_innovation == 2


def test_add_node_innovation_registers_split_connections_in_ledger():
    db = InnovationDB()
    ni = db.add_node_innovation(0, 10)
    # the two new wires are real inventions: later lookups reuse the numbers
    assert db.connection_innovation(0, ni.node_id) == ni.in_innovation
    assert db.connection_innovation(ni.node_id, 10) == ni.out_innovation


def test_add_node_innovation_reuses_on_same_split():
    db = InnovationDB()
    first = db.add_node_innovation(0, 10)
    second = db.add_node_innovation(0, 10)
    assert first == second


def test_add_node_innovation_different_splits_differ():
    db = InnovationDB()
    a = db.add_node_innovation(0, 10)
    b = db.add_node_innovation(1, 11)
    assert a.node_id != b.node_id
    assert a != b


def test_new_node_id_mints_fresh_ids():
    db = InnovationDB()
    assert db.new_node_id() == 14
    assert db.new_node_id() == 15


def test_recorded_node_innovation_lookup():
    db = InnovationDB()
    assert db.recorded_node_innovation(0, 10) is None
    innovation = db.add_node_innovation(0, 10)
    assert db.recorded_node_innovation(0, 10) == innovation


def test_serialization_round_trip_preserves_state():
    db = InnovationDB()
    db.connection_innovation(0, 10)
    db.connection_innovation(1, 11)
    db.add_node_innovation(0, 10)

    restored = InnovationDB.from_dict(db.to_dict())
    assert restored.connection_innovation(0, 10) == db.connection_innovation(0, 10)
    assert restored.add_node_innovation(0, 10) == db.add_node_innovation(0, 10)
    # counters continue identically on brand-new inventions
    assert restored.connection_innovation(9, 13) == db.connection_innovation(9, 13)
