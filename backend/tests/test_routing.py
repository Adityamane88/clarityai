from app.services.routing import choose_route


def test_choose_route_prefers_local_when_confident() -> None:
    decision = choose_route(
        user_message='Summarize the uploaded maintenance procedure.',
        local_confidence=0.42,
        local_hits=4,
        research_mode='auto',
    )
    assert decision.route == 'local'
    assert decision.needs_web_research is False


def test_choose_route_uses_research_for_current_info() -> None:
    decision = choose_route(
        user_message='What is the latest update on this regulation today?',
        local_confidence=0.05,
        local_hits=0,
        research_mode='auto',
    )
    assert decision.route == 'research'
    assert decision.needs_web_research is True
    assert decision.query_is_time_sensitive is True


def test_choose_route_can_force_hybrid() -> None:
    decision = choose_route(
        user_message='Compare my SOP with current best practices.',
        local_confidence=0.11,
        local_hits=2,
        research_mode='force',
    )
    assert decision.route == 'hybrid'
    assert decision.needs_web_research is True
    assert decision.needs_local_knowledge is True
