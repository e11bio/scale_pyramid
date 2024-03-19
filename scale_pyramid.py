from funlib.geometry import Coordinate, Roi
from funlib.persistence import prepare_ds, open_ds
from skimage.transform import rescale
import argparse
import daisy
import logging
import numpy as np
import zarr

logging.basicConfig(level=logging.INFO)


def downscale_block(in_array, out_array, factor, block):
    in_data = in_array.to_ndarray(block.read_roi, fill_value=0)

    if all(f == 1 for f in factor):
        logging.info(
            "Factor is all 1s, assuming in highest resolution, copying data over..."
        )
        out_data = in_data

    else:
        logging.info("Factor is not all 1s, downsampling data...")
        dims = len(factor)

        in_shape = Coordinate(in_data.shape[-dims:])
        assert in_shape.is_multiple_of(factor)

        n_channels = len(in_data.shape) - dims
        if n_channels >= 1:
            factor = (1,) * n_channels + factor

        factor = np.array((1,) * len(factor)) / np.array(factor)

        if in_data.dtype in (np.uint64, bool):
            order = 0
            anti_aliasing = False
        else:
            order = 1
            anti_aliasing = True

        out_data = rescale(
            in_data.astype(np.float32), factor, order=order, anti_aliasing=anti_aliasing
        ).astype(in_data.dtype)

    try:
        out_array[block.write_roi] = out_data
    except Exception:
        logging.info(f"Failed to write to {block.write_roi}")

        raise

    return 0


def downscale(in_array, out_array, factor, write_size, num_workers):
    logging.info(f"Downsampling by factor {factor}")

    dims = in_array.roi.dims
    block_roi = Roi((0,) * dims, write_size)

    logging.info(f"Processing ROI {out_array.roi} with blocks {block_roi}")

    downscale_task = daisy.Task(
        "downscale",
        out_array.roi,
        block_roi,
        block_roi,
        process_function=lambda b: downscale_block(in_array, out_array, factor, b),
        read_write_conflict=False,
        num_workers=num_workers,
        max_retries=0,
        fit="shrink",
    )

    done = daisy.run_blockwise([downscale_task])

    if not done:
        raise RuntimeError("Downscale task failed for (at least) one block")


def create_scale_pyramid(
    in_file,
    in_ds_name,
    scales,
    chunk_shape,
    num_workers=10,
    out_file=None,
    compressor={"id": "zstd", "level": 5},
):
    # make sure in_ds_name points to a dataset
    try:
        open_ds(in_file, in_ds_name)
    except Exception:
        raise RuntimeError(f"{in_ds_name} does not seem to be a dataset")

    if out_file is None:
        ds = zarr.open(in_file)

        if not in_ds_name.endswith("/s0"):
            ds_name = in_ds_name + "/s0"

            logging.info(f"Moving {in_ds_name} to {ds_name}")

            ds.store.rename(in_ds_name, in_ds_name + "__tmp")
            ds.store.rename(in_ds_name + "__tmp", ds_name)

        else:
            ds_name = in_ds_name
            in_ds_name = in_ds_name[:-3]
    else:
        ds_name = in_ds_name
        scales.insert(0, [1, 1, 1])  # so we copy over highest resolution

    logging.info(f"Scaling {in_file} by a factor of {scales}")

    prev_array = open_ds(in_file, ds_name)

    if chunk_shape is not None:
        chunk_shape = Coordinate(chunk_shape)
    else:
        chunk_shape = Coordinate(prev_array.data.chunks)
        logging.info(f"Reusing chunk shape of {chunk_shape} for new datasets")

    if prev_array.n_channel_dims == 0:
        num_channels = None
    elif prev_array.n_channel_dims == 1:
        num_channels = prev_array.shape[0]
    else:
        raise RuntimeError("more than one channel not yet implemented, sorry...")

    for scale_num, scale in enumerate(scales):
        try:
            scale = Coordinate(scale)
        except Exception:
            scale = Coordinate((scale,) * chunk_shape.dims)

        next_voxel_size = prev_array.voxel_size * scale
        next_total_roi = prev_array.roi.snap_to_grid(next_voxel_size, mode="grow")
        next_write_size = chunk_shape * next_voxel_size

        logging.info(f"Next voxel size: {next_voxel_size}")
        logging.info(f"Next total ROI: {next_total_roi}")
        logging.info(f"Next chunk size: {next_write_size}")

        next_ds = scale_num + 1 if out_file is None else scale_num
        next_ds_name = in_ds_name + "/s" + str(next_ds)

        logging.info(f"Preparing {next_ds_name}")

        next_array = prepare_ds(
            in_file if out_file is None else out_file,
            next_ds_name,
            total_roi=next_total_roi,
            voxel_size=next_voxel_size,
            write_size=next_write_size,
            dtype=prev_array.dtype,
            num_channels=num_channels,
            compressor=compressor,
        )

        downscale(prev_array, next_array, scale, next_write_size, num_workers)

        prev_array = next_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a scale pyramide for a zarr/N5 container."
    )

    parser.add_argument("--file", "-f", type=str, help="The input container")
    parser.add_argument("--ds", "-d", type=str, help="The name of the dataset")
    parser.add_argument(
        "--scales",
        "-s",
        nargs="*",
        type=int,
        required=True,
        help="The downscaling factor between scales",
    )
    parser.add_argument(
        "--chunk_shape",
        "-c",
        nargs="*",
        type=int,
        default=None,
        help="The size of a chunk in voxels",
    )

    args = parser.parse_args()

    create_scale_pyramid(args.file, args.ds, args.scales, args.chunk_shape)
