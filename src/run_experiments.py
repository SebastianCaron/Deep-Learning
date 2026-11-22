import argparse
import sys
from pathlib import Path

try:
    from src.ablation import run_ablation, ABLATION_CONFIGS
    from src.cross_validation import run_cross_validation
except ModuleNotFoundError:
    from ablation import run_ablation, ABLATION_CONFIGS
    from cross_validation import run_cross_validation


def main():
    parser = argparse.ArgumentParser(
        description="Campagne d'expérimentation pour ArtClassifierTinyBased (Ablation & Validation Croisée)"
    )
    parser.add_argument(
        "--task",
        choices=["ablation", "cross-validation", "all"],
        default="all",
        help="Type d'expérimentation à exécuter (ablation, cross-validation, ou all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Nombre maximum d'époques (défaut: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Taille du batch (défaut: 16)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Patience pour early stopping (défaut: 2)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Nombre de plis pour la validation croisée (défaut: 3)",
    )
    parser.add_argument(
        "--ablation-configs",
        nargs="+",
        default=["baseline", "no_batchnorm", "no_dropout", "no_latent_space", "frozen_backbone"],
        help="Configurations d'ablation à tester",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/Art Movement/",
        help="Chemin vers le dossier des données",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Dossier racine pour sauvegarder les checkpoints et résumés",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Taux d'apprentissage (learning rate)",
    )

    args = parser.parse_args()
    root_output = Path(args.output_dir)

    print("=" * 80)
    print("DÉMARRAGE DE LA SUITE D'EXPÉRIMENTATIONS : ArtClassifierTinyBased")
    print(f"Tâche : {args.task}")
    print(f"Hyperparamètres : Époques = {args.epochs} | Batch size = {args.batch_size} | Early Stopping Patience = {args.patience}")
    print("=" * 80)

    if args.task in ["ablation", "all"]:
        ablation_dir = root_output / "ablation"
        print("\n>>> LANCEMENT DE L'ÉTUDE D'ABLATION <<<")
        run_ablation(
            configs=args.ablation_configs,
            data_path=args.data_path,
            output_dir=ablation_dir,
            nb_epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            lr=args.lr,
        )

    if args.task in ["cross-validation", "all"]:
        cv_dir = root_output / "cross_val"
        print("\n>>> LANCEMENT DE LA VALIDATION CROISÉE <<<")
        run_cross_validation(
            n_splits=args.folds,
            data_path=args.data_path,
            output_dir=cv_dir,
            nb_epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            lr=args.lr,
        )

    print("\n" + "=" * 80)
    print("TOUTES LES EXPÉRIMENTATIONS DEMANDÉES SONT TERMINÉES.")
    print("=" * 80)


if __name__ == "__main__":
    main()
