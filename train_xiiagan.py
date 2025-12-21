import math
import time

import torch

from gimm.eval.fidelity import FrechetInceptionDistance, InceptionScore, KernelInceptionDistance, \
    PerceptualPathLength, PrecisionRecall
from gimm.logs.log import LoggerConsole, LoggerWandb, LoggerImageFile
from gimm.models.gap_x.xdcgan import DCGAN, ConfigIIA
from gimm.models.gap_aware.lsgan import LSGAN
from gimm.models.gap_aware.wgan import WGAN
from gimm.run.optimizer import Adam
from gimm.run.trainer import Trainer, TrainerConfig
from gimm.scheduler.constant import ConstantLR
from gimm.scheduler.exponential import ExponentialLR


# TODO: add continuous shuffling for dataset (seed += 1 for each rerun)

# Implement adam for classifier and adabelief for generator https://arxiv.org/pdf/2411.03999v1 (verify FID)
# Implement AdaBelief https://arxiv.org/pdf/2010.07468


def main(run_id: int = 0):
    # torch.set_float32_matmul_precision("medium")
    run_name = f"nsgan-base-{run_id}"
    config = TrainerConfig(
        project="xii-gan-length",
        group="nsgan-base",
        execution_name=run_name,
        batch_size=256,
        grad_accum_steps=1,
        steps_per_batch=1,
        steps_total=100_000, # 128 epochs
        # steps_total=20_000,  # 25.6 epochs
        steps_checkpoint=4_000_000,
        steps_eval=20_000,
        steps_log=1_000,
        steps_image=1_000,

        # d_updates_per_step=5,

        g_optimizer=Adam(lr=0.0002, betas=(0.5, 0.999)),
        d_optimizer=Adam(lr=0.0002, betas=(0.5, 0.999)),
        g_scheduler=ExponentialLR(gamma=0.001, last_step=3_906_250),
        d_scheduler=ExponentialLR(gamma=0.001, last_step=3_906_250),
        # d_scheduler=GapAwareLR(ideal_loss=math.log(4)),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        dataset="cifar10",
        dataset_dir="/media/main/Data/Datasets/gimm/",
        output_path=f"output/{run_name}",
        delete_checkpoint=True,

        validation_policy='supplement'
    )

    iia = ConfigIIA(
        iia_loss_weight=0.5,
        iia_start_step=110_000,
        iia_warmup_steps=10_000,
        iia_map_weighting=True,
        iia_maps='both',
        iia_maps_save_interval=20_000,
    )

    trainer = Trainer(
        model=DCGAN(iia_config=iia, output_path=config.output_path),
        configs=config,
        eval_metrics=[
            InceptionScore(),
            FrechetInceptionDistance(),
            KernelInceptionDistance(),
            PerceptualPathLength(),
            # PrecisionRecall()
        ],
        loggers=[
            LoggerConsole(),
            LoggerImageFile(path=config.output_path, img_format='PNG'),
            LoggerWandb(experiment=config.project, config=config.__dict__)
        ],
    )

    start_time = time.time()
    trainer.train()
    # trainer.train(DatasetMNIST(batch_size=128))
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    # result = trainer.evaluate()
    # print("Evaluation results:")
    # print(result)


if __name__ == "__main__":
    main(0)


# TODO: Overwrite checkpoint
