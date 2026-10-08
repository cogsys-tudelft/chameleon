import yaml

import numpy as np

from autolightning import auto_data

from torch_mate.data.utils import FewShot

from chameleon.core.numpy.learning import learn_with_few_shots
from chameleon.core.shared_utils import set_seed


ways = 20
shots = 5
query_shots = 10

net_path = '../../nets/omniglot_tcn_20_way_1_shot_acc=93.51.qsd.pkl'

with open("../../../meta-learning-arena/src/metalarena/configs/omniglot/few_shot_data.yaml") as f:
    data_args = yaml.safe_load(f)

data_args['data']['init_args'].update(
    dict(
        ways=ways,
        shots=shots,
        query_shots=query_shots,
    ))

ds = auto_data(data_args)
ds.setup('test')
ds_test = ds.get_transformed_dataset('test').dataset

set_seed(0)

ds_test = FewShot(ds_test,
                n_way=ways,
                k_shot=shots,
                query_shots=query_shots)

accs, preds_targets = learn_with_few_shots(net_path,
                                        iter(ds_test),
                                        shots=shots,
                                        ways=ways,
                                        weight_bit_width=4,
                                        query_shots=query_shots,
                                        num_repeats=10,
                                        slog2_weights=True,
                                        use_log2_fsl_weights=True)

mean_acc = np.mean(accs)
conf95 = 1.96 * np.std(accs) / np.sqrt(len(accs))

print(f"Accuracy over {len(accs)} steps: {mean_acc*100:.2f}% ± {conf95*100:.2f}% (95% CI) for {ways}-way {shots}-shot on Omniglot")
