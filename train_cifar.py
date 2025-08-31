import time

import torch

from gimm.datasets.mnist import DatasetMNIST
from gimm.datasets.toucan import DatasetToucan
from gimm.eval.fidelity import FrechetInceptionDistance, InceptionScore, KernelInceptionDistance, \
    PerceptualPathLength, PrecisionRecall
from gimm.logs.log import LoggerConsole, LoggerWandb, LoggerImageFile
from gimm.models.dcgan import DCGAN
from gimm.models.vitgan.vitgan import VitGAN
from gimm.run.optimizer import Adam
from gimm.run.trainer import Trainer, TrainerConfig
from gimm.scheduler.constant import ConstantLR
from gimm.scheduler.cosine import CosineLR
from gimm.scheduler.gap_aware import GapAwareLR


# TODO: add continuous shufflling for dataset (seed += 1 for each rerun)

# Implement adam for classifier and adabelief for generator https://arxiv.org/pdf/2411.03999v1 (verify FID)
# Implement AdaBelief https://arxiv.org/pdf/2010.07468


def main():
    # torch.set_float32_matmul_precision("medium")

    config = TrainerConfig(
        project="vitgan_cifar",
        batch_size=128,
        grad_accum_steps=1,
        steps_per_batch=1,
        steps_total=50_000,
        steps_checkpoint=1_000_000,
        steps_eval=50_000,
        steps_log=1_000,
        steps_image=1_000,

        g_optimizer=Adam(lr=0.00001, betas=(0.0, 0.999)),
        d_optimizer=Adam(lr=0.00001, betas=(0.0, 0.999)),
        g_scheduler=CosineLR(50_000),
        d_scheduler=CosineLR(50_000),
        # d_scheduler=GapAwareLR(),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        dataset="cifar10",
        dataset_dir="/media/main/Data/Datasets/gimm/",
        output_path="output_cifar_3",
    )

    trainer = Trainer(
        # model = VitGAN(),
        model = VitGAN(),
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
            # LoggerWandb(experiment=config.project, config=config.__dict__)
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
    main()
