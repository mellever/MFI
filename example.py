import itertools
import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import GoalProgrammingMixin, StateGoal
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.util import run_optimization_problem


class RangeGoal(StateGoal):
    def __init__(self, opt_prob, state, priority):
        self.state = state
        self.target_min = opt_prob.get_timeseries(state + "_min")
        self.target_max = opt_prob.get_timeseries(state + "_max")
        self.violation_timeseries_id = state + "_target_violation"
        self.function_value_timeseries_id = state
        self.priority = priority
        super().__init__(opt_prob)


class TargetGoal(StateGoal):
    def __init__(self, opt_prob, state, priority):
        self.state = state
        self.target_min = opt_prob.get_timeseries(state + "_target")
        self.target_max = self.target_min
        self.violation_timeseries_id = state + "_target_violation"
        self.function_value_timeseries_id = state
        self.priority = priority
        super().__init__(opt_prob)


class Example(
    HomotopyMixin,
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    channels = "LowerChannel", "MiddleChannel", "UpperChannel"
    channel_n_level_nodes = 2

    def pre(self):
        super().pre()
        # Generate handy tuples to iterate over
        self.channel_node_indices = tuple(range(1, self.channel_n_level_nodes + 1))
        self.channel_level_nodes = tuple(
            "{}.H[{}]".format(c, n)
            for c, n in itertools.product(self.channels, self.channel_node_indices)
        )
        # Expand channel water level goals to all nodes
        for channel in self.channels:
            channel_max = self.get_timeseries(channel + "_max")
            channel_min = self.get_timeseries(channel + "_min")
            for i in self.channel_node_indices:
                self.set_timeseries("{}.H[{}]_max".format(channel, i), channel_max)
                self.set_timeseries("{}.H[{}]_min".format(channel, i), channel_min)
        # Make input series appear in output csv
        self.set_timeseries("Inflow_Q", self.get_timeseries("Inflow_Q"))
        self.set_timeseries(
            "DrinkingWaterExtractionPump_Q_target",
            self.get_timeseries("DrinkingWaterExtractionPump_Q_target"),
        )
        self.io.set_timeseries('theta', self.io.datetimes, np.full_like(channel_max.values, 1.0))

    @property
    def h_th(self):
        # return self.parameters(0)['theta']
        if isinstance(self._HomotopyMixin__theta, float):
            return self.parameters(0)['theta']
        else:
            try:
                return self.io.get_timeseries('theta').values
            except:
                return self._HomotopyMixin__theta
            
    def homotopy_options(self):
        options = super().homotopy_options()
        number_of_linear_timesteps = 12
        options['non_linear_thresh_time_idx'] = len(self.io.datetimes)-number_of_linear_timesteps
        options['delta_theta_0'] = 0.5
        options['delta_theta_min'] = 0.010
        return options

    def parameters(self, ensemble_member):
        p = super().parameters(ensemble_member)
        times = self.times()
        p["step_size"] = times[1] - times[0]
        return p

    def path_goals(self):
        g = super().path_goals()

        # Add RangeGoal on water level states with a priority of 1
        for node in self.channel_level_nodes:
            g.append(RangeGoal(self, node, 1))

        # Add TargetGoal on Extraction Pump with a priority of 2
        g.append(TargetGoal(self, "DrinkingWaterExtractionPump_Q", 2))

        return g
    
    def priority_started(self, priority):
        super().priority_started(priority)

        self.priority = priority

        reduce_problems=True

        if reduce_problems:

            if isinstance(self.h_th, float) or isinstance(self.h_th, int):
                if self.h_th != 1.0:
                    priorities = [y[0] for y in self._GoalProgrammingMixin__subproblem_epsilons]
                    for i in priorities[1:]:
                        ii = [y[0] for y in self._GoalProgrammingMixin__subproblem_epsilons].index(i)
                        self._GoalProgrammingMixin__subproblem_epsilons.pop(ii)
                    return
            else:
                if self.h_th.any() != 1.0:
                    priorities = [y[0] for y in self._GoalProgrammingMixin__subproblem_epsilons]
                    for i in priorities[1:]:
                        ii = [y[0] for y in self._GoalProgrammingMixin__subproblem_epsilons].index(i)
                        self._GoalProgrammingMixin__subproblem_epsilons.pop(ii)
                    return

    def post(self):
        super().post()


# Run
import timeit
starttime = timeit.default_timer()
run_optimization_problem(Example)
print("RUNTIME :", timeit.default_timer() - starttime,"s")
