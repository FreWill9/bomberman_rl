import agent_code.agent_fred.helpers as fredhelpers

agent = self.world.active_agents[0]
state = self.world.get_state_for_agent(agent)
for x in range(self.world.arena.shape[1]):
    for y in range(self.world.arena.shape[0]):
        value = fredhelpers.closest_coin_dist(state, (x, y))
        self.render_text(str(value), s.GRID_OFFSET[0] + s.GRID_SIZE * (x + 0.5),
                         s.GRID_OFFSET[1] + s.GRID_SIZE * (y + 0.5),
                         (255, 255, 255),
                         valign='center', size='small')