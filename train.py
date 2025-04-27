import torch

from gimm.datasets.cifar10 import DatasetCifar10
from gimm.datasets.mnist import DatasetMNIST
from gimm.eval.fidelity import FrechetInceptionDistance, InceptionScore, KernelInceptionDistance, \
    PerceptualPathLength, PrecisionRecall
from gimm.logs.log import LoggerConsole, LoggerWandb, LoggerImageFile
from gimm.models.dcgan import DCGAN
from gimm.models.gan import GAN
from gimm.run.optimizer import Adam
from gimm.run.trainer import Trainer, TrainerConfig
from gimm.scheduler.constant import ConstantLR
from gimm.scheduler.gap_aware import GapAwareLR


# TODO: add continuous shufflling for dataset (seed += 1 for each rerun)

# Implement adam for classifier and adabelief for generator https://arxiv.org/pdf/2411.03999v1 (verify FID)
# Implement AdaBelief https://arxiv.org/pdf/2010.07468


def main():
    # torch.set_float32_matmul_precision("medium")

    config = TrainerConfig(
        project='test-dcgan',
        batch_size=128,
        grad_accum_steps=1,
        steps_per_batch=2,

        steps_total=200_000,
        steps_checkpoint=400_000,
        steps_eval=10_000,
        steps_log=1_000,
        steps_image=10_000,

        g_optimizer=Adam(lr=0.0001, betas=(0.5, 0.999)),
        d_optimizer=Adam(lr=0.0003, betas=(0.5, 0.999)),
        g_scheduler=ConstantLR(),
        d_scheduler=ConstantLR(),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DatasetCifar10(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory_device=device)
    # dataset = DatasetMNIST(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory_device=device)

    trainer = Trainer(
        # model = VitGAN()
        model = DCGAN(in_features=dataset.dims),
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
        ],)

    trainer.train(dataset)
    trainer.evaluate(dataset)


if __name__ == "__main__":
    main()
