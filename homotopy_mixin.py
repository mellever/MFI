import logging
from typing import Dict, Union

import numpy as np
import pandas as pd
from copy import copy

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class HomotopyMixin(OptimizationProblem):
    """
    Adds homotopy to your optimization problem.  A homotopy is a continuous transformation between
    two optimization problems, parametrized by a single parameter :math:`\\theta \\in [0, 1]`.

    Homotopy may be used to solve non-convex optimization problems, by starting with a convex
    approximation at :math:`\\theta = 0.0` and ending with the non-convex problem at
    :math:`\\theta = 1.0`.

    .. note::

        It is advised to look for convex reformulations of your problem, before resorting to a use
        of the (potentially expensive) homotopy process.

    """
    
    ### My implementation of the smart seed function
    def smartseed(self, seed):
        prev_result = "/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/results.csv"
        df = pd.read_csv(prev_result) #Read in the data from the previous results
        df.drop(columns=df.columns[0], axis=1, inplace=True) #Drop first column with the time
        dict_prev_result = df.to_dict('list') #Convert to dictionary

        #Assign the data from the results into the dictionary
        for key, result in dict_prev_result.items():
            times = self.times(key)
            times = times[1:] 
            result = result[1:]

            seed[key] = Timeseries(times, result)

        #Return the result
        return seed
    ### End of smart seed function

    def seed(self, ensemble_member):
        seed = super().seed(ensemble_member)
        options = self.homotopy_options()

        
        ### My addition to the code
        #If smartseed is true, go into the data to retrieve the seed from the last result. 
        ss = False
        if ss:
            smart_seed =  self.smartseed(seed)
            if self.__theta==1:
                compare_key_list = ["UpperChannel.H[1]", "UpperChannel.H[2]"]
                result_list_ss = [smart_seed[compare_key].values for compare_key in compare_key_list]

        ### End of my addition

        # Overwrite the seed only when the results of the latest run are
        # stored within this class. That is, when the GoalProgrammingMixin
        # class is not used or at the first run of the goal programming loop.
        overwrite_seed = False
        if isinstance(self.__theta, float) or isinstance(self.__theta, int):
            if self.__theta > options["theta_start"]:
                overwrite_seed = True
        else:
            if self.__theta.any() > options["theta_start"]:
                overwrite_seed = True
        if overwrite_seed and getattr(self, "_gp_first_run", True):
            for key, result in self.__results[ensemble_member].items():
                times = self.times(key)
                if (result.ndim == 1 and len(result) == len(times)) or (
                    result.ndim == 2 and result.shape[0] == len(times)
                ):
                    # Only include seed timeseries which are consistent
                    # with the specified time stamps.
                    seed[key] = Timeseries(times, result)

                elif (result.ndim == 1 and len(result) == 1) or (
                    result.ndim == 2 and result.shape[0] == 1
                ):
                    seed[key] = result
   
        #Addition
        if ss and self.__theta==1:
            result_list = np.array([seed[compare_key].values for compare_key in compare_key_list])
            diff_arr = []
            for j in range(len(compare_key_list)):
                res_ss = result_list_ss[j]
                res = result_list[j]
                diff = np.array([res_ss[i] - res[i] for i in range(len(res_ss))])
                diff_arr.append(diff)
            np.savetxt("/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/difference.txt", diff_arr)
            np.savetxt("/home/melle/Documents/Deltares/rtc-tools-examples/cascading_channels/output/keylist.txt", compare_key_list, fmt="%s")

        if ss: return smart_seed
        else: return seed
        ### Till here

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)

        options = self.homotopy_options()
        try:
            # Only set the theta if we are in the optimization loop. We want
            # to avoid accidental usage of the parameter value in e.g. pre().
            # Note that we use a try-except here instead of hasattr, to avoid
            # explicit name mangling.
            if isinstance(self.__theta, float) or isinstance(self.__theta, int):
                parameters[options["homotopy_parameter"]] = self.__theta
                self.io.set_timeseries(
                    options["homotopy_parameter"],
                    self.io.datetimes,
                    np.full_like(np.arange(len(self.io.datetimes), dtype=float), self.__theta),
                )
            else:
                parameters[options["homotopy_parameter"]] = self.__theta
                self.io.set_timeseries(
                    self.homotopy_options()["homotopy_parameter"],
                    self.io.datetimes,
                    self.__theta,
                )
        except AttributeError:
            pass

        return parameters

    def homotopy_options(self) -> Dict[str, Union[str, float]]:
        """
        Returns a dictionary of options controlling the homotopy process.

        +------------------------+------------+---------------+
        | Option                 | Type       | Default value |
        +========================+============+===============+
        | ``theta_start``        | ``float``  | ``0.0``       |
        +------------------------+------------+---------------+
        | ``delta_theta_0``      | ``float``  | ``1.0``       |
        +------------------------+------------+---------------+
        | ``delta_theta_min``    | ``float``  | ``0.01``      |
        +------------------------+------------+---------------+
        | ``homotopy_parameter`` | ``string`` | ``theta``     |
        +------------------------+------------+---------------+

        The homotopy process is controlled by the homotopy parameter in the model, specified by the
        option ``homotopy_parameter``.  The homotopy parameter is initialized to ``theta_start``,
        and increases to a value of ``1.0`` with a dynamically changing step size.  This step size
        is initialized with the value of the option ``delta_theta_0``.  If this step size is too
        large, i.e., if the problem with the increased homotopy parameter fails to converge, the
        step size is halved.  The process of halving terminates when the step size falls below the
        minimum value specified by the option ``delta_theta_min``.

        :returns: A dictionary of homotopy options.
        """

        return {
            "theta_start": 0.0,
            "delta_theta_0": 1.0,
            "delta_theta_min": 0.01,
            "homotopy_parameter": "theta",
            "non_linear_thresh_time_idx": len(self.io.datetimes),
        }

    def dynamic_parameters(self):
        dynamic_parameters = super().dynamic_parameters()

        if isinstance(self.__theta, float) or isinstance(self.__theta, int):
            if self.__theta > 0:
                # For theta = 0, we don't mark the homotopy parameter as being dynamic,
                # so that the correct sparsity structure is obtained for the linear model.
                options = self.homotopy_options()
                dynamic_parameters.append(self.variable(options["homotopy_parameter"]))
        else:
            if self.__theta.any() > 0:
                # For theta = 0, we don't mark the homotopy parameter as being dynamic,
                # so that the correct sparsity structure is obtained for the linear model.
                options = self.homotopy_options()
                dynamic_parameters.append(self.variable(options["homotopy_parameter"]))

        return dynamic_parameters

    def optimize(self, preprocessing=True, postprocessing=True, log_solver_failure_as_error=True):
        # Pre-processing
        if preprocessing:
            self.pre()

        options = self.homotopy_options()
        delta_theta = options["delta_theta_0"]

        do_theta_loop = False

        # Homotopy loop
        if options["non_linear_thresh_time_idx"] == len(self.io.datetimes):
            self.__theta = options["theta_start"]
            if self.__theta <= 1.0:
                do_theta_loop = True
        else:
            self.__theta = np.full_like(self.io.datetimes, float(options["theta_start"]))
            for i in range(options["non_linear_thresh_time_idx"], len(self.__theta)):
                self.__theta[i] = float(0.0)
            if self.__theta.any() <= 1.0:
                do_theta_loop = True

        while do_theta_loop:
            logger.info("Solving with homotopy parameter theta = {}.".format(self.__theta))

            success = super().optimize(
                preprocessing=False, postprocessing=False, log_solver_failure_as_error=False
            )
            if success:
                self.__results = [
                    self.extract_results(ensemble_member)
                    for ensemble_member in range(self.ensemble_size)
                ]
                if isinstance(self.__theta, float) or isinstance(self.__theta, int):
                    if self.__theta == 0.0:
                        self.check_collocation_linearity = False
                        self.linear_collocation = False

                        # Recompute the sparsity structure for the nonlinear model family.
                        self.clear_transcription_cache()
                else:
                    if self.__theta.any() == 0.0:
                        self.check_collocation_linearity = False
                        self.linear_collocation = False

                        # Recompute the sparsity structure for the nonlinear model family.
                        self.clear_transcription_cache()

            else:
                if isinstance(self.__theta, float) or isinstance(self.__theta, int):
                    if self.__theta == options["theta_start"]:
                        break
                else:
                    if (
                        self.__theta[0 : options["non_linear_thresh_time_idx"]].any()
                        == options["theta_start"]
                    ):
                        break
                if isinstance(self.__theta, float) or isinstance(self.__theta, int):
                    self.__theta -= delta_theta
                    delta_theta /= 2
                else:
                    for i in range(0, len(self.__theta)):
                        for i in range(0, options["non_linear_thresh_time_idx"]):
                            self.__theta[i] -= delta_theta
                            delta_theta /= 2

                if delta_theta < options["delta_theta_min"]:
                    failure_message = (
                        "Solver failed with homotopy parameter theta = {}. Theta cannot "
                        "be decreased further, as that would violate the minimum delta "
                        "theta of {}.".format(self.__theta, options["delta_theta_min"])
                    )
                    if log_solver_failure_as_error:
                        logger.error(failure_message)
                    else:
                        # In this case we expect some higher level process to deal
                        # with the solver failure, so we only log it as info here.
                        logger.info(failure_message)
                    break

            if isinstance(self.__theta, float) or isinstance(self.__theta, int):
                self.__theta += delta_theta
            else:
                for i in range(0, len(self.__theta)):
                    if i in range(0, options["non_linear_thresh_time_idx"]):
                        self.__theta[i] += delta_theta

            if isinstance(self.__theta, float) or isinstance(self.__theta, int):
                if self.__theta > 1.0:
                    do_theta_loop = False
            else:
                if self.__theta.any() > 1.0:
                    do_theta_loop = False

        # Post-processing
        if postprocessing:
            self.post()

        return success
