import sys
import mlflow
from sklearn.metrics import confusion_matrix, roc_auc_score
from config import COSTO_FP, COSTO_FN, PERCORSO_MLFLOW
from data import carica_e_pulisci, separa_X_y, dividi
from preprocessing import costruisci_preprocessore
from models import crea_modello, probabilita_positiva, allena
from threshold import scegli_soglia
from evaluation import report_soglia, report_test, report_importanze

MODELLO_DEFAULT = "logistic"

def main(nome_modello: str = MODELLO_DEFAULT):
    # 1-3. Dati: carico, pulisco, separo risposta e domande
    df = carica_e_pulisci()
    X, y = separa_X_y(df)
    # 4. Divido PRIMA di toccare i dati
    X_train, X_val, X_test, y_train, y_val, y_test = dividi(X, y)
    # 5. Preprocesso: il preprocessore impara categorie, media e deviazione DAL solo train
    preprocessore = costruisci_preprocessore(X)
    X_train = preprocessore.fit_transform(X_train)
    X_val = preprocessore.transform(X_val)
    X_test = preprocessore.transform(X_test)
    nomi_colonne = preprocessore.get_feature_names_out()

    mlflow.set_tracking_uri(PERCORSO_MLFLOW)
    mlflow.set_experiment("churn-prediction")
    with mlflow.start_run(run_name=nome_modello):
        # 6. Alleno il modello scelto dal registry
        print(f"Modello: {nome_modello}\n")
        model = crea_modello(nome_modello)
        allena(model, X_train, y_train, X_val, y_val)
        mlflow.log_params(model.get_params())

        # 7. Scelgo la soglia SUL VALIDATION
        prob_val = probabilita_positiva(model, X_val)
        soglia_scelta, griglia, costi = scegli_soglia(y_val, prob_val, COSTO_FP, COSTO_FN)
        mlflow.log_param("soglia", soglia_scelta)
        report_soglia(y_val, prob_val, soglia_scelta, griglia, costi, COSTO_FP, COSTO_FN)

        # 8. Test: una volta sola, con la soglia già decisa
        probabilita = probabilita_positiva(model, X_test)
        report_test(y_test, probabilita, soglia_scelta, COSTO_FP, COSTO_FN)

        # 9. Metriche per MLflow
        tn, fp, fn, tp = confusion_matrix(y_test, (probabilita >= soglia_scelta).astype(int)).ravel()
        mlflow.log_metric("auc_train", roc_auc_score(y_train, probabilita_positiva(model, X_train)))
        mlflow.log_metric("auc_test", roc_auc_score(y_test, probabilita))
        mlflow.log_metric("costo_test", fp * COSTO_FP + fn * COSTO_FN)
        mlflow.log_metric("recall_1", tp / (tp + fn))
        mlflow.log_metric("precision_1", tp / (tp + fp))

        # 10. Guardo dentro il modello
        report_importanze(model, nomi_colonne)

if __name__ == "__main__":
    nome = sys.argv[1] if len(sys.argv) > 1 else MODELLO_DEFAULT
    main(nome)
