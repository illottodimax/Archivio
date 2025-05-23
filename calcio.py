import pandas as pd
import numpy as np
# import requests # Non più strettamente necessario qui se non per altre funzionalità
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# import matplotlib.pyplot as plt # Commentato se non usato attivamente
from datetime import datetime, timedelta 

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import traceback 

# ==============================================================================
# FUNZIONI DI BASE (Alcune potrebbero non essere più usate o vanno adattate)
# ==============================================================================

# def scarica_dati(url, percorso_salvataggio): # NON PIÙ USATA DIRETTAMENTE QUI
#     """
#     Scarica i dati da un URL e li salva localmente.
#     Restituisce True in caso di successo, (False, errore_str) in caso di fallimento.
#     """
#     try:
#         response = requests.get(url, timeout=10) 
#         response.raise_for_status()
#         with open(percorso_salvataggio, 'wb') as file:
#             file.write(response.content)
#         return True
#     except requests.exceptions.RequestException as e: 
#         return False, f"Errore HTTP/Rete: {str(e)}"
#     except Exception as e:
#         return False, str(e)

def carica_dati_csv_globale(percorso_file): # Rinominata per chiarezza
    """
    Carica i dati delle partite da un singolo file CSV globale.
    Restituisce df in caso di successo, (None, errore_str) in caso di fallimento.
    """
    try:
        # Il CSV dallo scraper dovrebbe essere UTF-8
        df = pd.read_csv(percorso_file, encoding='utf-8', on_bad_lines='warn')
        if 'Date' in df.columns:
            # Lo script di scraping salva già le date in formato dd/mm/yy
            # Pandas dovrebbe interpretarle correttamente, ma forziamo il formato se necessario
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')
            except ValueError: # Se il formato è già datetime o un altro formato pandas-compatibile
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Conversione esplicita di FTHG e FTAG in numerico, gestendo i NaN che sono stringhe vuote o 'None'
        for col_score in ['FTHG', 'FTAG']:
            if col_score in df.columns:
                df[col_score] = pd.to_numeric(df[col_score], errors='coerce')
        return df
    except FileNotFoundError:
        return None, f"File non trovato: {percorso_file}"
    except pd.errors.EmptyDataError:
        return None, f"File vuoto: {percorso_file}"
    except Exception as e:
        return None, f"Errore caricamento CSV {percorso_file}: {str(e)}"

def pulisci_dati_csv_globale(df): # Rinominata e adattata
    """
    Pulisce i dati dal CSV globale e seleziona solo le colonne rilevanti.
    Il CSV da Livescore ha già una struttura definita.
    """
    if df is None or df.empty: 
        return pd.DataFrame()

    # Colonne attese dal CSV di Livescore
    colonne_attese = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    colonne_da_mantenere = [col for col in colonne_attese if col in df.columns]

    if not all(col in df.columns for col in ['Date', 'HomeTeam', 'AwayTeam']): # Colonne minime
         print("Attenzione: Colonne fondamentali (Date, HomeTeam, AwayTeam) mancanti nel CSV caricato.")
         # Potrebbe restituire df vuoto o gestire l'errore diversamente
         return pd.DataFrame() # O sollevare un'eccezione

    df_pulito = df[colonne_da_mantenere].copy()
    
    # Assicura che 'Date' sia datetime
    if 'Date' in df_pulito.columns and not pd.api.types.is_datetime64_any_dtype(df_pulito['Date']):
        try:
            df_pulito['Date'] = pd.to_datetime(df_pulito['Date'], format='%d/%m/%y', errors='coerce')
        except ValueError:
             df_pulito['Date'] = pd.to_datetime(df_pulito['Date'], errors='coerce')


    # Per FTR, i valori mancanti potrebbero essere stringhe vuote o 'None' dallo scraping se non gestiti prima
    # Sostituisci stringhe vuote o la stringa 'None' con np.nan reale per FTR
    if 'FTR' in df_pulito.columns:
        df_pulito['FTR'] = df_pulito['FTR'].replace(['', 'None'], np.nan)


    # Dropna per le righe dove FTR è NaN ma FTHG/FTAG non lo sono (partite passate senza risultato registrato)
    # e per righe con informazioni base mancanti.
    # Per l'addestramento, FTR è cruciale. Per le previsioni, FTR sarà NaN.
    # Questa pulizia è più mirata all'addestramento.
    subset_dropna_addestramento = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']
    colonne_presenti_per_dropna = [col for col in subset_dropna_addestramento if col in df_pulito.columns]
    
    # Rimuovi righe solo se TUTTE le colonne chiave per un risultato giocato sono mancanti
    # o se FTR è mancante per una partita che DOVREBBE avere un risultato.
    # Questa logica potrebbe essere rivista. Per ora, un dropna più semplice su info base:
    df_pulito.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'], inplace=True) # Info base devono esserci

    return df_pulito


def prepara_caratteristiche(df, n_partite_precedenti=5):
    # Questa funzione dovrebbe rimanere in gran parte invariata
    # Assicura che df in input abbia 'Date', 'HomeTeam', 'AwayTeam', 'FTR' (può essere NaN per future)
    if df.empty: 
        return pd.DataFrame()
    df_prep = df.copy()

    if 'FTR' in df_prep.columns and df_prep['FTR'].notna().any():
        idx_not_na_ftr = df_prep['FTR'].notna()
        df_prep.loc[idx_not_na_ftr, 'ResultValue'] = df_prep.loc[idx_not_na_ftr, 'FTR'].map({'H': 1, 'D': 0, 'A': -1}).astype(float) # Assicura float
    else:
        df_prep['ResultValue'] = np.nan 

    # Assicurati che Date sia ordinabile
    if 'Date' not in df_prep.columns or not pd.api.types.is_datetime64_any_dtype(df_prep['Date']):
        # print("Colonna 'Date' non è di tipo datetime o mancante in prepara_caratteristiche.")
        return pd.DataFrame() # Non possiamo procedere senza date ordinate
    df_prep.sort_values('Date', inplace=True)

    squadre = pd.concat([df_prep['HomeTeam'], df_prep['AwayTeam']]).dropna().unique()
    if not squadre.any(): return df_prep

    stats_squadre = {squadra: {'FormaRecente': []} for squadra in squadre}
    
    df_prep['FormaCasa'] = 0.0 # Inizializza come float
    df_prep['FormaTrasferta'] = 0.0 # Inizializza come float

    for i, row in df_prep.iterrows():
        ht = row['HomeTeam']
        at = row['AwayTeam']

        if pd.notna(ht) and ht in stats_squadre:
            forma_casa_val = stats_squadre[ht]['FormaRecente']
            df_prep.loc[i, 'FormaCasa'] = np.mean(forma_casa_val) if forma_casa_val else 0.0
        
        if pd.notna(at) and at in stats_squadre:
            forma_trasf_val = stats_squadre[at]['FormaRecente']
            df_prep.loc[i, 'FormaTrasferta'] = np.mean(forma_trasf_val) if forma_trasf_val else 0.0

        if pd.notna(row.get('ResultValue')): # Usa .get per sicurezza, anche se l'abbiamo creato
            rv = row['ResultValue']
            if pd.notna(ht) and ht in stats_squadre:
                if rv == 1: stats_squadre[ht]['FormaRecente'].append(3)
                elif rv == 0: stats_squadre[ht]['FormaRecente'].append(1)
                else: stats_squadre[ht]['FormaRecente'].append(0)
                stats_squadre[ht]['FormaRecente'] = stats_squadre[ht]['FormaRecente'][-n_partite_precedenti:]
            
            if pd.notna(at) and at in stats_squadre:
                if rv == -1: stats_squadre[at]['FormaRecente'].append(3)
                elif rv == 0: stats_squadre[at]['FormaRecente'].append(1)
                else: stats_squadre[at]['FormaRecente'].append(0)
                stats_squadre[at]['FormaRecente'] = stats_squadre[at]['FormaRecente'][-n_partite_precedenti:]
    return df_prep


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

def run_analysis_logic(output_widget, soglia_conf_var, orizzonte_gg_var): # Rimosso urls_to_process_dict
    def log_message(message):
        if output_widget:
            output_widget.config(state=tk.NORMAL)
            output_widget.insert(tk.END, str(message) + "\n")
            output_widget.see(tk.END) 
            output_widget.config(state=tk.DISABLED)

    try:
        log_message("Avvio analisi...")
        soglia_confidenza_gui = float(soglia_conf_var.get())
        orizzonte_giorni_gui = int(orizzonte_gg_var.get())
        log_message(f"Soglia confidenza: {soglia_confidenza_gui}, Orizzonte previsione: {orizzonte_giorni_gui} giorni")

        PERCORSO_CSV_GLOBALE = "dati_livescore_tutti_campionati.csv" 

        log_message(f"\n--- FASE 1: Caricamento Dati da {PERCORSO_CSV_GLOBALE} ---")
        
        if not os.path.exists(PERCORSO_CSV_GLOBALE):
            log_message(f"ERRORE: File dati '{PERCORSO_CSV_GLOBALE}' non trovato.")
            log_message("Eseguire prima lo script di scraping per generare questo file.")
            messagebox.showerror("Errore File", f"File dati '{PERCORSO_CSV_GLOBALE}' non trovato. Eseguire prima lo script di scraping.")
            return

        caricamento_status = carica_dati_csv_globale(PERCORSO_CSV_GLOBALE) 
        
        if isinstance(caricamento_status, tuple) and caricamento_status[0] is None:
            log_message(f"  -> ERRORE caricamento {PERCORSO_CSV_GLOBALE}: {caricamento_status[1]}")
            messagebox.showerror("Errore Caricamento", f"Errore caricando {PERCORSO_CSV_GLOBALE}: {caricamento_status[1]}")
            return
        
        df_input = caricamento_status

        if df_input is None or df_input.empty:
            log_message(f"  -> File {PERCORSO_CSV_GLOBALE} non caricato o vuoto.")
            messagebox.showerror("Errore Dati", f"File {PERCORSO_CSV_GLOBALE} non caricato o vuoto.")
            return
        else:
            log_message(f"  -> File {PERCORSO_CSV_GLOBALE} caricato. Righe iniziali: {len(df_input)}")

        log_message("\n--- FASE 2: Pulizia Dati ---")
        df_combinato = pulisci_dati_csv_globale(df_input.copy()) # Lavora su una copia
        if df_combinato.empty:
           log_message("Errore: DataFrame vuoto dopo la pulizia iniziale.")
           messagebox.showerror("Errore Dati", "DataFrame vuoto dopo la pulizia iniziale.")
           return
        log_message(f"Dati dopo pulizia base. Righe: {len(df_combinato)}")

        # Mappatura 'Div' a 'Campionato'
        mappa_div_campionato = {
            "I1": "SerieA", "I2": "SerieB", "SP1": "LaLiga", "F1": "Ligue1",
            "D1": "Bundesliga", "E0": "PremierLeague", "ITC1": "CoppaItalia"
            # Aggiungi altre mappature se i tuoi 'Div' sono diversi/più numerosi
        }
        if 'Div' in df_combinato.columns:
            df_combinato['Campionato'] = df_combinato['Div'].map(mappa_div_campionato).fillna(df_combinato['Div'])
        else:
            log_message("ATTENZIONE: Colonna 'Div' non trovata per creare 'Campionato'. Usare un placeholder o gestire.")
            df_combinato['Campionato'] = "Sconosciuto" 

        log_message(f"\n--- Dataset Combinato Pronto ---")
        log_message(f"Dataset combinato (da file unico) pronto. Shape: {df_combinato.shape}")
        
        if 'Date' in df_combinato.columns:
            df_combinato.dropna(subset=['Date'], inplace=True) # Assicura che non ci siano Date NaN
            df_combinato.sort_values(by='Date', inplace=True)
            if not df_combinato.empty:
                log_message(f"Data minima: {df_combinato['Date'].min().strftime('%Y-%m-%d')}, Data massima: {df_combinato['Date'].max().strftime('%Y-%m-%d')}")
            else: log_message("df_combinato vuoto dopo dropna su Date o ordinamento.")
        else:
            log_message("ERRORE: Colonna 'Date' mancante in df_combinato."); return
        
        if df_combinato.empty: log_message("ERRORE: df_combinato vuoto prima di prepara_caratteristiche."); return

        log_message("\n--- FASE 3: Preparazione Caratteristiche Avanzate ---")
        df_features_complete = prepara_caratteristiche(df_combinato.copy()) 
        
        if df_features_complete.empty:
             log_message("Errore: df_features_complete è vuoto dopo prepara_caratteristiche."); return
        log_message(f"Dati dopo preparazione caratteristiche. Righe: {len(df_features_complete)}")


        oggi = pd.to_datetime(datetime.now().date())
        df_addestramento_candidati = df_features_complete[
            (df_features_complete['Date'] < oggi) & 
            (df_features_complete['FTR'].notna()) & 
            (df_features_complete['FTHG'].notna()) & 
            (df_features_complete['FTAG'].notna())
        ].copy()

        data_limite_previsione = oggi + pd.to_timedelta(orizzonte_giorni_gui, unit='d')
        df_da_prevedere_candidati = df_features_complete[
            (df_features_complete['Date'] >= oggi) &
            (df_features_complete['Date'] < data_limite_previsione)
        ].copy()
        
        log_message(f"\n--- Divisione Dati ---")
        log_message(f"Dati addestramento candidati (partite < {oggi.strftime('%Y-%m-%d')} con FTR e punteggi): {len(df_addestramento_candidati)} partite")
        log_message(f"Dati da prevedere candidati (partite da {oggi.strftime('%Y-%m-%d')} a {data_limite_previsione.strftime('%Y-%m-%d')}): {len(df_da_prevedere_candidati)} partite")

        if df_addestramento_candidati.empty:
            log_message("Nessun dato di addestramento disponibile dopo i filtri."); return
        
        # === MODIFICA PER CARATTERISTICHE MODELLO ===
        caratteristiche_modello = ['FormaCasa', 'FormaTrasferta'] # Feature numeriche di base

        # Aggiungiamo 'Div' come feature categorica. 
        # La funzione crea_modello_predittivo applicherà pd.get_dummies()
        # Puoi scegliere 'Div' (codice es. I1, SP1) o 'Campionato' (nome es. SerieA, LaLiga)
        # 'Div' è solitamente più conciso.
        colonna_campionato_da_usare = 'Div' 

        if colonna_campionato_da_usare in df_addestramento_candidati.columns:
            # Verifica se la colonna ha più di un valore unico prima di aggiungerla,
            # altrimenti pd.get_dummies con drop_first=True potrebbe rimuoverla completamente se c'è un solo valore.
            if df_addestramento_candidati[colonna_campionato_da_usare].nunique() > 1:
                caratteristiche_modello.append(colonna_campionato_da_usare)
                log_message(f"Aggiunta '{colonna_campionato_da_usare}' alle caratteristiche del modello.")
            else:
                log_message(f"ATTENZIONE: Colonna '{colonna_campionato_da_usare}' ha un solo valore unico, non utile per la dummificazione con drop_first=True.")
        else:
            log_message(f"ATTENZIONE: Colonna '{colonna_campionato_da_usare}' non trovata nei dati di addestramento, non verrà usata come feature.")
        # === FINE MODIFICA PER CARATTERISTICHE MODELLO ===

        df_addestramento_final = df_addestramento_candidati.dropna(subset=['ResultValue'] + [col for col in caratteristiche_modello if col in df_addestramento_candidati.columns]).copy() # Assicura che le feature esistano
        log_message(f"Dati addestramento finali (dopo dropna su ResultValue e features): {len(df_addestramento_final)} partite")


        if df_addestramento_final.empty or not caratteristiche_modello: # Verifica se caratteristiche_modello non è vuota
            log_message("Dati di addestramento insufficienti o nessuna caratteristica valida selezionata."); return
            
        # Rimuovi ulteriormente le caratteristiche che potrebbero non esistere più dopo il dropna precedente
        caratteristiche_modello_effettive = [col for col in caratteristiche_modello if col in df_addestramento_final.columns]
        if not caratteristiche_modello_effettive:
            log_message("Nessuna caratteristica valida rimasta dopo la pulizia finale dei dati di addestramento."); return

        log_message(f"Caratteristiche per il modello: {caratteristiche_modello_effettive}")
        log_message("\n--- FASE 4: Addestramento Modello ---")
        model, X_columns_trained_final = crea_modello_predittivo(df_addestramento_final, caratteristiche_modello_effettive, log_func=log_message)
        
        if model is None:
            log_message("Addestramento modello fallito."); return
        log_message("Modello addestrato.")

        log_message("\n--- FASE 5: Suggerimento Partite ---")
        if not df_da_prevedere_candidati.empty:
            log_message(f"Suggerimenti con soglia >= {soglia_confidenza_gui}:")
            partite_consigliate_df = suggerisci_partite(df_da_prevedere_candidati.copy(), model, X_columns_trained_final, soglia_confidenza=soglia_confidenza_gui)
            
            if not partite_consigliate_df.empty:
                max_len_home = partite_consigliate_df['HomeTeam'].astype(str).map(len).max()
                max_len_away = partite_consigliate_df['AwayTeam'].astype(str).map(len).max()
                max_len_prev = partite_consigliate_df['PrevisioneLabel'].astype(str).map(len).max()
                
                max_len_home = max(10, int(max_len_home) if pd.notna(max_len_home) else 10) + 2
                max_len_away = max(10, int(max_len_away) if pd.notna(max_len_away) else 10) + 2
                max_len_prev = max(15, int(max_len_prev) if pd.notna(max_len_prev) else 15) + 2

                header = f"{'Date':<11} {'HomeTeam':<{max_len_home}} {'AwayTeam':<{max_len_away}} {'Previsione':<{max_len_prev}} MaxProb"
                log_message("Partite Consigliate:\n" + header)
                log_message("-" * len(header))
                for _, row_sugg in partite_consigliate_df[['Date', 'HomeTeam', 'AwayTeam', 'PrevisioneLabel', 'MaxProb']].head(30).iterrows():
                    log_message(f"{pd.to_datetime(row_sugg['Date']).strftime('%d/%m/%y'):<11} {str(row_sugg['HomeTeam']):<{max_len_home}} {str(row_sugg['AwayTeam']):<{max_len_away}} {str(row_sugg['PrevisioneLabel']):<{max_len_prev}} {row_sugg['MaxProb']:.2f}")
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
    
    # urls_config non è più passata a run_analysis_logic
    analysis_thread = threading.Thread(target=run_analysis_logic, 
                                       args=(output_text, soglia_var, orizzonte_var))
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