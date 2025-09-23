import argparse
import tensorflow as tf
import logging
import traceback
import os
import numpy as np

from common import (
    get_eeg_embedding_model,
    get_ppg_embedding_model,
    get_bimodal_model,
    TripletModel,
    load_data,
    create_triplet_dataset_bimodal,
)

def main():
    parser = argparse.ArgumentParser(description="Treinamento Local")
    parser.add_argument("--fusion", type=str, default="concatenacao",
                        choices=["concatenacao", "soma", "media"],
                        help="Estratégia de fusão dos embeddings EEG + PPG")
    parser.add_argument("--eeg-path", type=str, required=True, help="Caminho para os arquivos EEG")
    parser.add_argument("--ppg-path", type=str, required=True, help="Caminho para os arquivos PPG")
    args = parser.parse_args()

    # Arquivo de log
    log_file = "local_error.log"
    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        logging.info("Client script started.")

        # Carregar dados
        logging.info("Loading data...")
        X_train_eeg, y_train, X_test_eeg, y_test = load_data("eeg", data_path_prefix=args.eeg_path)
        X_train_ppg, _, X_test_ppg, _ = load_data("ppg", data_path_prefix=args.ppg_path)

        usuarios = np.unique(y_train)
        if len(usuarios) < 3:
            raise ValueError("O dataset tem menos de 3 usuários!")
        usuarios_selecionados = usuarios[:3]
        logging.info(f"Treinando para usuários: {usuarios_selecionados}")

         # Corte para o menor tamanho (Nao sei se pode fazer isso!!!!)
        min_len = min(len(X_train_eeg), len(X_train_ppg), len(y_train))
        X_train_eeg = X_train_eeg[:min_len]
        X_train_ppg = X_train_ppg[:min_len]
        y_train = y_train[:min_len]

        min_len_test = min(len(X_test_eeg), len(X_test_ppg), len(y_test))
        X_test_eeg = X_test_eeg[:min_len_test]
        X_test_ppg = X_test_ppg[:min_len_test]
        y_test = y_test[:min_len_test]

        # Filtrar dados para apenas os 3 usuários selecionados
        mask_train = np.isin(y_train, usuarios_selecionados)
        X_train_eeg = X_train_eeg[mask_train]
        X_train_ppg = X_train_ppg[mask_train]
        y_train = y_train[mask_train]

        mask_test = np.isin(y_test, usuarios_selecionados)
        X_test_eeg = X_test_eeg[mask_test]
        X_test_ppg = X_test_ppg[mask_test]
        y_test = y_test[mask_test]

        logging.info(f"Treino EEG: {len(X_train_eeg)}, PPG: {len(X_train_ppg)}")
        logging.info(f"Teste EEG: {len(X_test_eeg)}, PPG: {len(X_test_ppg)}")

        if len(X_train_eeg) == 0 or len(X_train_ppg) == 0:
            print(f"No training data found for selected users. Exiting gracefully.")
            logging.warning("No training data found for selected users. Exiting gracefully.")
            return

        logging.info("Creating triplet dataset...")
        train_dataset = create_triplet_dataset_bimodal(X_train_eeg, X_train_ppg, y_train, batch_size=64)
        test_dataset = create_triplet_dataset_bimodal(X_test_eeg, X_test_ppg, y_test, batch_size=64)
        logging.info("Triplet dataset created.")

        logging.info("Inicializing model...")
        embedding_model = get_bimodal_model(embedding_dim=128, fusion=args.fusion)
        model = TripletModel(embedding_model, margin=0.5)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        logging.info("Modelo compilado.")

        model.fit(train_dataset, epochs=10, verbose=1)
        print("Treinamento concluído.")

    
        loss, _ = model.evaluate(test_dataset, verbose=1)
        print(f"Avaliação final - Loss: {loss}")
        logging.info(f"Avaliação concluída. Loss: {loss}")

        model.save_weights("modelo_bimodal.h5")
        logging.info("Pesos do modelo salvos.")

    except BaseException as e:
        print(f"An error occurred on client for model. See {log_file} for details.")
        logging.error("Client crashed with an exception: \n%s", traceback.format_exc())

if __name__ == "__main__":
    main()