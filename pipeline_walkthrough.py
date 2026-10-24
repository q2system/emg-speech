"""A short, plotted tour of the EMG pipeline.

This is the file students read first.  It starts from the examples already
saved by the acquisition app and calls one helper function per pipeline stage.
The longer functions and all Matplotlib setup live in ``helper_lib/``.

Run:

    python pipeline_walkthrough.py --data dataset_demo --epochs 3
    python pipeline_walkthrough.py --data dataset --save-dir plots

Each section opens one figure.  The helper library saves the figures instead
when a display is not available or when ``--save-dir`` is used.
"""

from __future__ import annotations

import sys

from _bootstrap import add_src_to_path

add_src_to_path()

from helper_lib import (
    build_model,
    compute_loss_and_gradients,
    evaluate_model,
    forward_pass,
    goodbye,
    load_database,
    make_batch,
    make_plotter,
    parse_args,
    plot_batch,
    plot_database,
    plot_evaluation,
    plot_forward,
    plot_loss,
    plot_model_architecture,
    plot_preprocessing,
    plot_training,
    prepare_data,
    train_model,
    welcome,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    welcome()

    try:
        plotter = make_plotter(args.save_dir, args.no_show)

        # =====================================================================
        # SECTION 1 - PRE-PROCESSING: THE SAVED DATABASE
        # =====================================================================
        index = load_database(args.data)
        plot_database(index, plotter)

        # =====================================================================
        # SECTION 2 - PRE-PROCESSING: SPLIT AND NORMALISE
        # =====================================================================
        prepared = prepare_data(
            index,
            val_fraction=args.val_fraction,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        plot_preprocessing(prepared, plotter)

        # =====================================================================
        # SECTION 3 - BATCHING: THE MODEL INPUT
        # =====================================================================
        batch = make_batch(prepared)
        plot_batch(batch, plotter)

        # =====================================================================
        # SECTION 4 - THE MODEL: ARCHITECTURE
        # =====================================================================
        bundle = build_model(
            prepared,
            device_name=args.device,
            learning_rate=args.learning_rate,
        )
        plot_model_architecture(bundle, batch, plotter)

        # =====================================================================
        # SECTION 5 - THE MODEL: A FORWARD PASS
        # =====================================================================
        forward = forward_pass(bundle, batch)
        plot_forward(forward, plotter)

        # =====================================================================
        # SECTION 6 - LOSS AND BACKPROPAGATION
        # =====================================================================
        losses = compute_loss_and_gradients(bundle, batch)
        plot_loss(losses, plotter)

        # =====================================================================
        # SECTION 7 - TRAINING LOOP
        # =====================================================================
        training = train_model(bundle, prepared, epochs=args.epochs)
        plot_training(training, plotter)

        # =====================================================================
        # SECTION 8 - TRAINING RESULTS: CONFUSION MATRIX AND METRICS
        # =====================================================================
        evaluation = evaluate_model(training)
        plot_evaluation(evaluation, plotter)

    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Walkthrough failed: {exc}", file=sys.stderr)
        return 1

    goodbye()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
