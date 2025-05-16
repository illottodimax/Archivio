import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
# import seaborn as sns # Commentato se non usato attivamente
from datetime import datetime, timedelta # Aggiunto timedelta

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import traceback # Per stampare il traceback completo degli errori

# ==============================================================================
# FUNZIONI DI BASE (Le tue funzioni)
# ==============================================================================

def scarica_dati(url, percorso_salvataggio):
    """
    Scarica i dati da un URL e li salva localmente.
    Restituisce True in caso di successo, (False, errore_str) in caso di fallimento.
    """
    try:
        response = requests.get(url, timeout=10) # Aggiunto timeout
        response.raise_for_status()
        with open(percorso_salvataggio, 'wb') as file:
            file.write(response.content)
        return True
    except requests.exceptions.RequestException as e: # Più specifico per errori di rete/http
        return False, f"Errore HTTP/Rete: {str(e)}"
    except Exception as e:
        return False, str(e)

def carica_dati(percorso_file):
    """
    Carica i dati delle partite da un file CSV.
    Restituisce df in caso di successo, (None, errore_str) in caso di fallimento.
    """
    try:
        # Prova con encoding comuni, latin1 è spesso usato per questi file
        df = pd.read_csv(percorso_file, encoding='latin1', on_bad_lines='warn') # 'warn' invece di 'skip' per vedere se ci sono problemi
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        return df
    except FileNotFoundError:
        return None, "File non trovato."
    except pd.errors.EmptyDataError:
        return None, "File vuoto."
    except Exception as e:
        return None, str(e)

def pulisci_dati(df):
    """
    Pulisce i dati e seleziona solo le colonne rilevanti.
    """
    if df is None or df.empty: 
        return pd.DataFrame() # Restituisci DataFrame vuoto se input è None o vuoto

    colonne_necessarie = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    colonne_presenti_df = df.columns.tolist()

    colonne_da_mantenere = [col for col in colonne_necessarie if col in colonne_presenti_df]
    
    colonne_quote = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA'] # Espandi se necessario
    colonne_da_mantenere.extend([col for col in colonne_quote if col in colonne_presenti_df])
    
    altre_statistiche = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    colonne_da_mantenere.extend([col for col in altre_statistiche if col in colonne_presenti_df])
    
    colonne_da_mantenere = sorted(list(set(colonne_da_mantenere))) # Rimuovi duplicati e ordina
    
    if not any(col in colonne_da_mantenere for col in colonne_necessarie if col != 'Date'): # Se mancano colonne chiave oltre la data
         # Potresti loggare un avviso più specifico qui
         pass

    if not colonne_da_mantenere: # Se nessuna colonna è stata selezionata
        return pd.DataFrame()

    df_pulito = df[colonne_da_mantenere].copy()
    
    # Colonne indispensabili per il dropna per la logica successiva (es. calcolo ResultValue)
    subset_dropna_fondamentali = [col for col in ['Date', 'HomeTeam', 'AwayTeam', 'FTR'] if col in df_pulito.columns]
    if len(subset_dropna_fondamentali) == 4: # Solo se tutte e 4 sono presenti
        df_pulito.dropna(subset=subset_dropna_fondamentali, inplace=True)
    else: # Se mancano colonne fondamentali, il df potrebbe non essere utilizzabile per feature complesse
        # print("Attenzione: colonne fondamentali per FTR mancanti, la pulizia potrebbe essere incompleta.")
        pass
        
    return df_pulito


def prepara_caratteristiche(df, n_partite_precedenti=5):
    """
    Crea caratteristiche avanzate per il modello predittivo.
    """
    if df.empty: 
        return pd.DataFrame()
    df_prep = df.copy()

    # 'ResultValue' può essere calcolato solo se 'FTR' è presente e non NaN
    if 'FTR' in df_prep.columns and df_prep['FTR'].notna().any():
        # Applica la mappatura solo dove FTR non è NaN per evitare errori con tipi misti se ci sono NaN
        idx_not_na_ftr = df_prep['FTR'].notna()
        df_prep.loc[idx_not_na_ftr, 'ResultValue'] = df_prep.loc[idx_not_na_ftr, 'FTR'].map({'H': 1, 'D': 0, 'A': -1})
    else:
        # Se FTR manca o è tutto NaN, non possiamo creare ResultValue per la forma
        # Assegniamo NaN, così le partite future non influenzeranno la forma calcolata
        df_prep['ResultValue'] = np.nan 

    squadre_home = df_prep['HomeTeam'].dropna().unique() if 'HomeTeam' in df_prep.columns else []
    squadre_away = df_prep['AwayTeam'].dropna().unique() if 'AwayTeam' in df_prep.columns else []
    squadre = list(set(list(squadre_home)) | set(list(squadre_away)))

    if not squadre: 
        return df_prep # Restituisce il df così com'è se non ci sono squadre (improbabile)

    stats_squadre_per_feature_creation = {squadra: {'FormaRecente': []} for squadra in squadre}

    if 'Date' not in df_prep.columns: 
        return df_prep # Impossibile ordinare e calcolare forma correttamente
    df_prep.sort_values('Date', inplace=True)

    features_list = []
    for i, partita in df_prep.iterrows():
        home_team = partita['HomeTeam']
        away_team = partita['AwayTeam']
        
        feature_row = {}
        
        # Copia tutte le colonne esistenti dal df_prep originale a feature_row
        for col in df_prep.columns:
            feature_row[col] = partita[col]

        if pd.notna(home_team) and home_team in stats_squadre_per_feature_creation:
            current_forma_casa = stats_squadre_per_feature_creation[home_team]['FormaRecente']
            feature_row['FormaCasa'] = np.mean(current_forma_casa) if current_forma_casa else 0
        else:
            feature_row['FormaCasa'] = 0 # Default se squadra non valida

        if pd.notna(away_team) and away_team in stats_squadre_per_feature_creation:
            current_forma_trasferta = stats_squadre_per_feature_creation[away_team]['FormaRecente']
            feature_row['FormaTrasferta'] = np.mean(current_forma_trasferta) if current_forma_trasferta else 0
        else:
            feature_row['FormaTrasferta'] = 0 # Default

        features_list.append(feature_row)
        
        risultato_val_partita = partita.get('ResultValue', np.nan) # Usa .get per sicurezza
        if pd.notna(risultato_val_partita): # Aggiorna la forma solo se il risultato è noto
            if pd.notna(home_team) and home_team in stats_squadre_per_feature_creation:
                if risultato_val_partita == 1: stats_squadre_per_feature_creation[home_team]['FormaRecente'].append(3)
                elif risultato_val_partita == 0: stats_squadre_per_feature_creation[home_team]['FormaRecente'].append(1)
                else: stats_squadre_per_feature_creation[home_team]['FormaRecente'].append(0) # Include sconfitta (-1)
                stats_squadre_per_feature_creation[home_team]['FormaRecente'] = \
                    stats_squadre_per_feature_creation[home_team]['FormaRecente'][-n_partite_precedenti:]

            if pd.notna(away_team) and away_team in stats_squadre_per_feature_creation:
                if risultato_val_partita == -1: stats_squadre_per_feature_creation[away_team]['FormaRecente'].append(3) # Vittoria trasferta
                elif risultato_val_partita == 0: stats_squadre_per_feature_creation[away_team]['FormaRecente'].append(1) # Pareggio
                else: stats_squadre_per_feature_creation[away_team]['FormaRecente'].append(0) # Sconfitta trasferta (Vittoria casa = 1)
                stats_squadre_per_feature_creation[away_team]['FormaRecente'] = \
                    stats_squadre_per_feature_creation[away_team]['FormaRecente'][-n_partite_precedenti:]
    
    if not features_list: 
        return pd.DataFrame(columns=df_prep.columns.tolist() + ['FormaCasa', 'FormaTrasferta']) # Schema colonne
    return pd.DataFrame(features_list)


def separa_dati_recenti(df, n_partite=10): # Questa funzione potrebbe non essere più necessaria con il nuovo approccio
    if df.empty or len(df) <= n_partite :
        return df, pd.DataFrame(columns=df.columns) 
    df_ordinato = df.sort_values('Date')
    df_storico = df_ordinato.iloc[:-n_partite]
    df_recenti = df_ordinato.iloc[-n_partite:]
    return df_storico, df_recenti


def prepara_dati_per_previsione(df_previsione, X_columns_trained_model):
    # df_previsione qui dovrebbe contenere solo le feature, non Date, HomeTeam, AwayTeam, FTR, ResultValue, etc.
    if df_previsione.empty: 
        return pd.DataFrame(columns=X_columns_trained_model) 

    # Applica get_dummies solo sulle colonne che sono effettivamente categoriche e presenti
    # X_columns_trained_model contiene già i nomi delle colonne dopo il get_dummies del training
    # Quindi, dobbiamo creare le dummy e poi allineare.
    
    # Seleziona le colonne originali (prima di get_dummies) che erano nel training
    # Questo è un punto debole: dovremmo sapere quali erano le categoriche originali.
    # Per ora, assumiamo che df_previsione abbia le stesse colonne *originali* del set di training.
    
    X_prev = df_previsione.copy()

    # Riempire NaN nelle feature numeriche prima di get_dummies
    for col in X_prev.select_dtypes(include=np.number).columns:
        if col in X_columns_trained_model: # Solo se era una feature usata
             X_prev[col] = X_prev[col].fillna(0) # O altra strategia di imputation

    X_prev = pd.get_dummies(X_prev, drop_first=True) 
    
    for col in X_columns_trained_model:
        if col not in X_prev.columns:
            X_prev[col] = 0 # Aggiungi colonne dummy mancanti (es. una categoria non presente in questo subset)
            
    return X_prev[X_columns_trained_model] # Assicura ordine e presenza colonne


def suggerisci_partite(df_future_with_all_info, model, X_columns_trained_model, soglia_confidenza=0.6):
    cols_attese_output = ['Date', 'HomeTeam', 'AwayTeam', 'PrevisioneLabel', 'MaxProb',
                          'Prob_VittoriaCasa', 'Prob_Pareggio', 'Prob_VittoriaTrasferta']
    if df_future_with_all_info.empty:
        return pd.DataFrame(columns=cols_attese_output)

    # Estrai solo le feature necessarie per la predizione, basandoti su X_columns_trained_model
    # X_columns_trained_model sono i nomi delle colonne DOPO get_dummies
    # df_future_with_all_info ha le colonne ORIGINALI più quelle calcolate (FormaCasa, etc.)
    
    # Seleziona le colonne originali che, una volta trasformate, produrranno X_columns_trained_model
    # Questo è il passaggio più complicato per allineare. È più facile se `prepara_dati_per_previsione`
    # prende le colonne originali e fa il get_dummies al suo interno.
    
    # Creiamo un subset di df_future_with_all_info con le colonne grezze che corrispondono
    # a quelle in X_columns_trained_model (prima del dummify) + quelle numeriche già presenti.
    # Esempio semplificato: assumiamo che X_columns_trained_model includa nomi originali per numeriche
    # e nomi dummificati per categoriche.
    
    df_features_for_prediction = df_future_with_all_info.copy()

    X_prev_ready = prepara_dati_per_previsione(df_features_for_prediction, X_columns_trained_model)
    
    if X_prev_ready.empty:
        return pd.DataFrame(columns=cols_attese_output)

    try:
        y_pred = model.predict(X_prev_ready)
        probabilita = model.predict_proba(X_prev_ready)
    except Exception as e:
        # log_func(f"Errore durante model.predict/predict_proba: {e}")
        return pd.DataFrame(columns=cols_attese_output)

    df_risultati = df_future_with_all_info.copy() # Inizia con tutte le info originali
    df_risultati['PrevisioneRisultato'] = y_pred
    df_risultati['PrevisioneLabel'] = df_risultati['PrevisioneRisultato'].map({1: 'Vittoria Casa', 0: 'Pareggio', -1: 'Vittoria Trasferta'})
    
    class_map = {val: i for i, val in enumerate(model.classes_)}
    
    # Assegna le probabilità con cautela, gestendo le classi mancanti
    # Se una classe non è in class_map (perché non era nel training o y_train aveva una sola classe),
    # la sua probabilità sarà 0.
    idx_casa = class_map.get(1)
    idx_pareggio = class_map.get(0)
    idx_trasferta = class_map.get(-1)

    df_risultati['Prob_VittoriaCasa'] = probabilita[:, idx_casa] if idx_casa is not None and idx_casa < probabilita.shape[1] else 0
    df_risultati['Prob_Pareggio'] = probabilita[:, idx_pareggio] if idx_pareggio is not None and idx_pareggio < probabilita.shape[1] else 0
    df_risultati['Prob_VittoriaTrasferta'] = probabilita[:, idx_trasferta] if idx_trasferta is not None and idx_trasferta < probabilita.shape[1] else 0
    
    # MaxProb calcolato correttamente sulle probabilità disponibili
    prob_cols_for_max = []
    if 'Prob_VittoriaCasa' in df_risultati: prob_cols_for_max.append('Prob_VittoriaCasa')
    if 'Prob_Pareggio' in df_risultati: prob_cols_for_max.append('Prob_Pareggio')
    if 'Prob_VittoriaTrasferta' in df_risultati: prob_cols_for_max.append('Prob_VittoriaTrasferta')
    
    if prob_cols_for_max:
         df_risultati['MaxProb'] = df_risultati[prob_cols_for_max].max(axis=1)
    else:
         df_risultati['MaxProb'] = 0

    partite_consigliate = df_risultati[df_risultati['MaxProb'] >= soglia_confidenza].copy()
    partite_consigliate = partite_consigliate.sort_values('MaxProb', ascending=False)
    
    final_cols_present = [col for col in cols_attese_output if col in partite_consigliate.columns]
    return partite_consigliate[final_cols_present]


def crea_modello_predittivo(df_train_input, caratteristiche_input, log_func=print):
    df = df_train_input.copy() # Lavora su una copia
    caratteristiche = list(caratteristiche_input) # Lavora su una copia

    if df.empty or not caratteristiche:
        log_func("Dati di addestramento o caratteristiche mancanti per crea_modello_predittivo.")
        return None, []
        
    # Assicurati che tutte le caratteristiche selezionate esistano nel DataFrame
    caratteristiche = [col for col in caratteristiche if col in df.columns]
    if not caratteristiche:
        log_func("Nessuna delle caratteristiche specificate è presente nel DataFrame di addestramento.")
        return None, []

    X = df[caratteristiche]
    
    if 'ResultValue' not in df.columns or df['ResultValue'].isnull().all():
        log_func("Colonna target 'ResultValue' mancante o tutta NaN in crea_modello_predittivo.")
        return None, []
    y = df['ResultValue'].dropna() # Rimuovi NaN da y e allinea X
    X = X.loc[y.index] # Allinea X con y dopo dropna

    # Riempire NaN nelle feature numeriche PRIMA di get_dummies
    for col in X.select_dtypes(include=np.number).columns:
        X[col] = X[col].fillna(X[col].median()) # Usa mediana o media per imputation
    # Per le categoriche, get_dummies gestirà i NaN creando una colonna apposita se non vengono rimossi
    # Oppure, se preferisci, X.dropna(subset=X.select_dtypes(include='object').columns, inplace=True) e riallinea y.

    try:
        X_processed = pd.get_dummies(X, drop_first=True, dummy_na=False) # dummy_na=False per non creare colonne per NaN
    except Exception as e:
        log_func(f"Errore durante pd.get_dummies: {e}")
        return None, []
    
    # Riallinea y con X_processed se get_dummies ha cambiato gli indici (improbabile con dummy_na=False se non ci sono NaN)
    y = y.loc[X_processed.index]

    if X_processed.empty or y.empty or len(X_processed) != len(y):
        log_func(f"Dati X ({X_processed.shape}) o y ({y.shape}) vuoti o non allineati dopo pre-elaborazione.")
        return None, []

    if len(y.unique()) < 2:
        log_func(f"Target y ha {len(y.unique())} classi uniche. Impossibile addestrare un classificatore binario/multiclasse.")
        log_func(f"Distribuzione di y: {y.value_counts().to_dict()}")
        return None, X_processed.columns

    try:
        # Usa stratify se possibile, altrimenti no.
        stratify_option = y if len(y.unique()) > 1 and y.value_counts().min() >= 2 else None # Stratify solo se ci sono almeno 2 campioni per classe
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=stratify_option)
    except Exception as e_split: # Cattura errore più generico
        log_func(f"Errore in train_test_split (fallback): {e_split}. Tento senza stratify.")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        except Exception as e_split_fatal:
            log_func(f"Errore fatale in train_test_split: {e_split_fatal}")
            return None, []

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample' if stratify_option is not None else None)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        log_func(f"Errore durante model.fit: {e}")
        return None, X_processed.columns

    if not hasattr(model, 'classes_') or len(model.classes_) < 2:
        log_func("Modello non addestrato correttamente o con meno di 2 classi (in model.classes_).")
        log_func(f"Classi in y_train effettive: {np.unique(y_train)}")
        return None, X_processed.columns

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    unique_labels_report = np.unique(np.concatenate((y_test.astype(str), y_pred.astype(str))))
    
    try:
        report = classification_report(y_test, y_pred, labels=model.classes_, target_names=[str(c) for c in model.classes_], zero_division=0)
    except Exception as e_report: # Più generico
        log_func(f"Errore nella generazione del classification_report: {e_report}")
        report = "Impossibile generare il report dettagliato."

    log_func(f"Accuratezza del modello (su test set storico): {accuracy:.4f}")
    log_func("Report di classificazione (su test set storico):")
    log_func(report)
    
    try:
        importance = pd.DataFrame({
            'Caratteristica': X_processed.columns, 
            'Importanza': model.feature_importances_
        }).sort_values('Importanza', ascending=False)
        
        log_func("\nImportanza delle caratteristiche (prime 10):")
        for _, row_importance in importance.head(10).iterrows():
            log_func(f"  {row_importance['Caratteristica']:<30} {row_importance['Importanza']:.4f}")
    except Exception as e_importance:
        log_func(f"Errore nel calcolo/stampa dell'importanza delle feature: {e_importance}")
            
    return model, X_processed.columns


# ==============================================================================
# LOGICA GUI E FUNZIONE PRINCIPALE run_analysis_logic
# ==============================================================================

def run_analysis_logic(output_widget, soglia_conf_var, orizzonte_gg_var, urls_to_process_dict):
    """Contiene la logica principale del tuo programma, modificata per output su GUI."""
    def log_message(message):
        if output_widget:
            output_widget.config(state=tk.NORMAL)
            output_widget.insert(tk.END, str(message) + "\n") # Converti a stringa per sicurezza
            output_widget.see(tk.END) 
            output_widget.config(state=tk.DISABLED)
            # output_widget.update_idletasks() # Rimosso per performance, il mainloop di Tkinter dovrebbe gestire l'aggiornamento

    try:
        log_message("Avvio analisi...")
        soglia_confidenza_gui = float(soglia_conf_var.get())
        orizzonte_giorni_gui = int(orizzonte_gg_var.get())
        log_message(f"Soglia confidenza: {soglia_confidenza_gui}, Orizzonte previsione: {orizzonte_giorni_gui} giorni")

        os.makedirs('dati_calcio', exist_ok=True)
        
        dataframes_raw_dict = {} 
        log_message("\n--- FASE 1: Download e Caricamento Dati ---")
        
        for nome_file_key, url_value in urls_to_process_dict.items():
            percorso = f"dati_calcio/{nome_file_key}.csv" # Usa la chiave del dizionario per il nome file
            log_message(f"Processando: {nome_file_key}")
            
            download_status = scarica_dati(url_value, percorso)
            if isinstance(download_status, tuple) and not download_status[0]: 
                 log_message(f"  -> ERRORE download {nome_file_key}: {download_status[1]}")
                 continue
            elif not download_status: 
                 log_message(f"  -> ERRORE download sconosciuto {nome_file_key}.")
                 continue
            else:
                 log_message(f"  -> File {nome_file_key} scaricato/aggiornato.")

            caricamento_status = carica_dati(percorso)
            if isinstance(caricamento_status, tuple) and caricamento_status[0] is None:
                log_message(f"  -> ERRORE caricamento {nome_file_key}: {caricamento_status[1]}")
                continue
            df_singolo_caricato = caricamento_status

            if df_singolo_caricato is not None and not df_singolo_caricato.empty:
                if 'Date' in df_singolo_caricato.columns and df_singolo_caricato['Date'].notna().any():
                    # Assicurati che le date valide siano presenti
                    df_valido_con_date = df_singolo_caricato.dropna(subset=['Date'])
                    if not df_valido_con_date.empty:
                        log_message(f"  -> {len(df_valido_con_date)} righe. Date da {df_valido_con_date['Date'].min().strftime('%Y-%m-%d')} a {df_valido_con_date['Date'].max().strftime('%Y-%m-%d')}")
                        dataframes_raw_dict[nome_file_key] = df_valido_con_date
                    else:
                        log_message(f"  -> File {nome_file_key}: vuoto dopo rimozione date non valide.")
                else:
                    log_message(f"  -> File {nome_file_key}: Colonna 'Date' mancante o tutta NaN.")
            else:
                log_message(f"  -> File {nome_file_key}: Non caricato o vuoto dopo `carica_dati`.")
        
        lista_df_da_combinare = []
        for nome_origine_file, df_contenuto in dataframes_raw_dict.items():
            df_processato = pulisci_dati(df_contenuto)
            if df_processato is not None and not df_processato.empty:
                df_processato['FonteFile'] = nome_origine_file 
                df_processato['Campionato'] = nome_origine_file.split('_')[0]
                lista_df_da_combinare.append(df_processato)

        if not lista_df_da_combinare:
            log_message("\nNessun dato valido da nessun file dopo la pulizia. Uscita.")
            messagebox.showerror("Errore", "Nessun dato valido caricato dopo la pulizia.")
            return

        df_combinato = pd.concat(lista_df_da_combinare, ignore_index=True)
        log_message(f"\n--- FASE 2: Dataset Combinato ---")
        log_message(f"Dataset combinato creato. Shape: {df_combinato.shape}")

        if df_combinato.empty:
            log_message("Errore: df_combinato è vuoto.")
            messagebox.showerror("Errore", "df_combinato vuoto.")
            return
        
        if 'Date' in df_combinato.columns:
            df_combinato.sort_values(by='Date', inplace=True)
            if not df_combinato.empty:
                log_message(f"Data massima: {df_combinato['Date'].max().strftime('%Y-%m-%d')}")
                log_message(f"Data minima: {df_combinato['Date'].min().strftime('%Y-%m-%d')}")
            else:
                log_message("df_combinato vuoto dopo ordinamento (improbabile se non era vuoto prima).")
        else:
            log_message("Errore: Colonna 'Date' mancante in df_combinato.")
            messagebox.showerror("Errore", "Colonna 'Date' mancante.")
            return
        
        log_message("\n--- FASE 3: Preparazione Caratteristiche ---")
        df_features_complete = prepara_caratteristiche(df_combinato) # Passa df_combinato
        
        if df_features_complete.empty:
             log_message("Errore: df_features_complete è vuoto dopo prepara_caratteristiche.")
             messagebox.showerror("Errore", "Errore nella preparazione delle caratteristiche.")
             return

        oggi = pd.to_datetime(datetime.now().date()) # Usa .date() per mezzanotte
       
        # Partite per addestramento: quelle passate (< oggi) e con un risultato FTR valido
        df_addestramento_candidati = df_features_complete[
            (df_features_complete['Date'] < oggi) & 
            (df_features_complete['FTR'].notna())
        ].copy()

        # Partite da prevedere: da oggi in avanti per N giorni
        data_limite_previsione = oggi + pd.to_timedelta(orizzonte_giorni_gui, unit='d')
        df_da_prevedere_candidati = df_features_complete[
            (df_features_complete['Date'] >= oggi) &
            (df_features_complete['Date'] < data_limite_previsione)
        ].copy()
        
        log_message(f"\n--- Divisione Dati ---")
        log_message(f"Dati addestramento (partite < {oggi.strftime('%Y-%m-%d')} con FTR): {df_addestramento_candidati.shape[0]} partite")
        log_message(f"Dati da prevedere (partite da {oggi.strftime('%Y-%m-%d')} a {data_limite_previsione.strftime('%Y-%m-%d')}): {df_da_prevedere_candidati.shape[0]} partite")

        if df_addestramento_candidati.empty:
            log_message("Nessun dato di addestramento disponibile.")
            messagebox.showwarning("Attenzione", "Nessun dato storico per addestrare il modello.")
            return
        
        # Definizione delle caratteristiche
        caratteristiche_da_escludere = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'ResultValue', 'Campionato', 'FonteFile']
        caratteristiche_modello = [col for col in df_addestramento_candidati.columns if col not in caratteristiche_da_escludere]
        caratteristiche_modello = [col for col in caratteristiche_modello if df_addestramento_candidati[col].notna().any()] # Rimuovi colonne tutte NaN

        # Drop NaN solo sul set di addestramento e solo per le feature selezionate
        # È cruciale che ResultValue non sia NaN qui. FTR.notna() dovrebbe averlo già gestito.
        df_addestramento_final = df_addestramento_candidati.dropna(subset=caratteristiche_modello + ['ResultValue']).copy()


        if df_addestramento_final.empty:
            log_message("Dati di addestramento vuoti dopo dropna sulle caratteristiche selezionate e ResultValue.")
            messagebox.showerror("Errore", "Dati di addestramento insufficienti dopo pulizia NaN.")
            return

        if not caratteristiche_modello:
            log_message("Nessuna caratteristica valida per il modello.")
            messagebox.showerror("Errore", "Nessuna caratteristica per il modello.")
            return
            
        log_message(f"Numero di caratteristiche per il modello: {len(caratteristiche_modello)}")
        log_message("\n--- FASE 4: Addestramento Modello ---")
        model, X_columns_trained_final = crea_modello_predittivo(df_addestramento_final, caratteristiche_modello, log_func=log_message)
        
        if model is None:
            log_message("Addestramento modello fallito.")
            messagebox.showerror("Errore", "Addestramento modello fallito.")
            return
        log_message("Modello addestrato.")

        log_message("\n--- FASE 5: Suggerimento Partite ---")
        if not df_da_prevedere_candidati.empty:
            log_message(f"Suggerimenti con soglia >= {soglia_confidenza_gui}:")
            
            # Per le partite da prevedere, le quote potrebbero mancare. `prepara_dati_per_previsione`
            # dovrebbe riempire con 0 le colonne mancanti rispetto a X_columns_trained_final
            partite_consigliate_df = suggerisci_partite(df_da_prevedere_candidati, model, X_columns_trained_final, soglia_confidenza=soglia_confidenza_gui)
            
            if not partite_consigliate_df.empty:
                partite_consigliate_display_df = partite_consigliate_df.copy()
                if 'Date' in partite_consigliate_display_df.columns:
                     partite_consigliate_display_df['Date'] = pd.to_datetime(partite_consigliate_display_df['Date']).dt.strftime('%Y-%m-%d')
                
                log_message("Partite Consigliate:")
                # Formattazione per allineamento colonne
                # Calcola la larghezza massima per HomeTeam e AwayTeam per un migliore allineamento
                max_len_home = partite_consigliate_display_df['HomeTeam'].astype(str).map(len).max() + 2
                max_len_away = partite_consigliate_display_df['AwayTeam'].astype(str).map(len).max() + 2
                max_len_prev = partite_consigliate_display_df['PrevisioneLabel'].astype(str).map(len).max() + 2

                header = f"{'Date':<11} {'HomeTeam':<{max_len_home}} {'AwayTeam':<{max_len_away}} {'Previsione':<{max_len_prev}} MaxProb"
                log_message(header)
                log_message("-" * len(header))

                for _, row_sugg in partite_consigliate_display_df[['Date', 'HomeTeam', 'AwayTeam', 'PrevisioneLabel', 'MaxProb']].head(30).iterrows():
                    log_message(f"{row_sugg['Date']:<11} {str(row_sugg['HomeTeam']):<{max_len_home}} {str(row_sugg['AwayTeam']):<{max_len_away}} {str(row_sugg['PrevisioneLabel']):<{max_len_prev}} {row_sugg['MaxProb']:.2f}")
            else:
                log_message(f"Nessuna partita consigliata con soglia >= {soglia_confidenza_gui}.")
        else:
            log_message("Nessuna partita futura (nel range specificato) da prevedere.")
            
        log_message("\nAnalisi completata.")
        messagebox.showinfo("Completato", "Analisi completata con successo!")

    except Exception as e_main_logic:
        log_message(f"ERRORE CRITICO DURANTE L'ANALISI: {e_main_logic}")
        log_message(traceback.format_exc())
        messagebox.showerror("Errore Critico", f"Si è verificato un errore imprevisto: {e_main_logic}")
    finally:
        if output_widget:
            output_widget.config(state=tk.NORMAL)


def start_analysis_thread_wrapper():
    run_button.config(state=tk.DISABLED)
    output_text.config(state=tk.NORMAL)
    output_text.delete('1.0', tk.END)
    output_text.config(state=tk.DISABLED)
    
    analysis_thread = threading.Thread(target=run_analysis_logic, 
                                       args=(output_text, soglia_var, orizzonte_var, urls_config))
    analysis_thread.daemon = True 
    analysis_thread.start()
    check_thread_status(analysis_thread)

def check_thread_status(thread):
    if thread.is_alive():
        root.after(200, lambda: check_thread_status(thread))
    else:
        run_button.config(state=tk.NORMAL)
        if output_text: # Controlla se output_text esiste ancora (in caso di chiusura GUI)
            output_text.config(state=tk.NORMAL)
        # messagebox.showinfo("Info", "Processo terminato.") # Rimosso per evitare pop-up se l'utente ha già visto "Completato"


# ==============================================================================
# CONFIGURAZIONE URLS E GUI
# ==============================================================================

urls_config = {
    # --- SOLO STAGIONE 2024/2025 (COMPLETA CON TUTTI I TUOI URL) ---
    "SerieA_2425": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "SerieB_2425": "https://www.football-data.co.uk/mmz4281/2425/I2.csv",
    "Bundesliga_2425": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "Bundesliga2_2425": "https://www.football-data.co.uk/mmz4281/2425/D2.csv",
    "LaLiga_2425": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "LaLiga2_2425": "https://www.football-data.co.uk/mmz4281/2425/SP2.csv",
    "Ligue1_2425": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
    "Ligue2_2425": "https://www.football-data.co.uk/mmz4281/2425/F2.csv",
    "PremierLeague_2425": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "Championship_2425": "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
    "League1_2425": "https://www.football-data.co.uk/mmz4281/2425/E2.csv",
    "League2_2425": "https://www.football-data.co.uk/mmz4281/2425/E3.csv",
    "Conference_2425": "https://www.football-data.co.uk/mmz4281/2425/EC.csv",
    "Eredivisie_2425": "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
    "JupilerLeague_2425": "https://www.football-data.co.uk/mmz4281/2425/B1.csv",
    "PrimeiraLiga_2425": "https://www.football-data.co.uk/mmz4281/2425/P1.csv",
    "SuperLig_2425": "https://www.football-data.co.uk/mmz4281/2425/T1.csv",
    "SuperLeague_2425": "https://www.football-data.co.uk/mmz4281/2425/G1.csv",
    "ScottishPL_2425": "https://www.football-data.co.uk/mmz4281/2425/SC0.csv", 
    "ScottishCH_2425": "https://www.football-data.co.uk/mmz4281/2425/SC1.csv", 
    "ScottishL1_2425": "https://www.football-data.co.uk/mmz4281/2425/SC2.csv", 
    "ScottishL2_2425": "https://www.football-data.co.uk/mmz4281/2425/SC3.csv", 
    "Usa_2425": "https://www.football-data.co.uk/new/USA.csv",
    "Svizzera_2425": "https://www.football-data.co.uk/new/SWZ.csv",
    "Svezia_2425": "https://www.football-data.co.uk/new/SWE.csv",
    "Russia_2425": "https://www.football-data.co.uk/new/RUS.csv",
    "Romania_2425": "https://www.football-data.co.uk/new/ROU.csv",
    "Polonia_2425": "https://www.football-data.co.uk/new/POL.csv",
    "Norvegia_2425": "https://www.football-data.co.uk/new/NOR.csv",
    "Messico_2425": "https://www.football-data.co.uk/new/MEX.csv",
    "Giappone_2425": "https://www.football-data.co.uk/new/JPN.csv",
    "Irlanda_2425": "https://www.football-data.co.uk/new/IRL.csv",
    "Finlandia_2425": "https://www.football-data.co.uk/new/FIN.csv",
    "Danimarca_2425": "https://www.football-data.co.uk/new/DNK.csv",
    "Cina_2425": "https://www.football-data.co.uk/new/CHN.csv",
    "Brasile_2425": "https://www.football-data.co.uk/new/BRA.csv",
    "Argentina_2425": "https://www.football-data.co.uk/new/ARG.csv",
    "Austria_2425": "https://www.football-data.co.uk/new/AUT.csv",
}

root = tk.Tk()
root.title("Analizzatore Partite Calcio v0.2") # Versione aggiornata
root.geometry("950x750") # Leggermente più grande

main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

controls_frame = ttk.LabelFrame(main_frame, text="Controlli", padding="10")
controls_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

soglia_label = ttk.Label(controls_frame, text="Soglia Confidenza (0.0-1.0):")
soglia_label.pack(side=tk.LEFT, padx=(0, 5))
soglia_var = tk.StringVar(value="0.55")
soglia_entry = ttk.Entry(controls_frame, textvariable=soglia_var, width=5)
soglia_entry.pack(side=tk.LEFT, padx=5)

orizzonte_label = ttk.Label(controls_frame, text="Orizzonte Previsione (giorni):")
orizzonte_label.pack(side=tk.LEFT, padx=(10, 5))
orizzonte_var = tk.StringVar(value="14")
orizzonte_entry = ttk.Entry(controls_frame, textvariable=orizzonte_var, width=4)
orizzonte_entry.pack(side=tk.LEFT, padx=5)

run_button = ttk.Button(controls_frame, text="Esegui Analisi", command=start_analysis_thread_wrapper)
run_button.pack(side=tk.LEFT, padx=20, pady=5)

output_frame = ttk.LabelFrame(main_frame, text="Log e Risultati", padding="10")
output_frame.pack(fill=tk.BOTH, expand=True, pady=5)

output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Courier New", 9), state=tk.DISABLED, height=25)
output_text.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root.mainloop()