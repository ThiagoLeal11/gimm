import torch

from gimm.eval.fidelity import FrechetInceptionDistance, InceptionScore, KernelInceptionDistance, \
    PerceptualPathLength, PrecisionRecall
from gimm.logs.log import LoggerConsole, LoggerWandb, LoggerImageFile
from gimm.models.dcgan import DCGAN
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
        project='dcgan',
        batch_size=64,
        grad_accum_steps=1,
        steps_per_batch=1,

        steps_total=500_000,
        steps_checkpoint=800_000,
        steps_eval=50_000,
        steps_log=1_000,
        steps_image=10_000,

        g_optimizer=Adam(lr=0.0002, betas=(0.5, 0.999)),
        d_optimizer=Adam(lr=0.0002, betas=(0.5, 0.999)),
        g_scheduler=ConstantLR(),
        d_scheduler=GapAwareLR(),
        # d_scheduler=GapAwareLR(x_min=0.1386, x_max=0.1386),

        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        dataset='cifar10',
        dataset_dir='/media/main/Data/Datasets/gimm/',
        output_path='output1'
    )

    trainer = Trainer(
        # model = VitGAN()
        model = DCGAN(),
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

    trainer.train()
    result = trainer.evaluate()
    print("Evaluation results:")
    print(result)


if __name__ == "__main__":
    main()
