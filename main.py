from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path

from pipeline.config import load_config
from pipeline.data_bundle import build_data_bundle
from pipeline.runner import build_runner

from src.utils import set_seed, setup_logger


def main():
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()

    # settings
    config_file = Path(args.config_file)
    config = load_config(config_file)

    output_dir = Path(f'./output/{config_file.stem}/')
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    logger = getLogger(__name__)
    setup_logger(logger, output_dir / 'log.txt')

    # load data
    data_bundle = build_data_bundle(config)

    # train
    runner = build_runner(config)
    runner.run(data_bundle)

    # save file
    runner.save(output_dir)


if __name__ == '__main__':
    main()
