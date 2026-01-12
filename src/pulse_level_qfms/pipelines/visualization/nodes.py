from pulse_level_qfms.utils import create_time_domain_viz

import logging

log = logging.getLogger(__name__)


def visualize_time_domain(model, train_loader, noise_params):
    if model.n_input_feat == 1:
        fig = create_time_domain_viz(model, train_loader, noise_params)
    else:
        raise NotImplementedError("Only 1D time domain visualization is supported")

    return {"figure": fig}
