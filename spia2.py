# -*- coding: utf-8 -*-
# Versione 3.8 - RITORNO AL FUNZIONANTE + FIX VARI (Incluso IndentationError)

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import pandas as pd
import numpy as np
import os
from tkcalendar import DateEntry
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
import traceback # Per debug
import sys # Per controllo modalità interattiva
import itertools # Importato per le combinazioni

# Prova a importare seaborn, ma continua anche se non è disponibile
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Variabili globali
risultati_globali = []
info_ricerca_globale = {}
file_ruote = {}

# =============================================================================
# FUNZIONI GRAFICHE
# =============================================================================
def crea_grafico_barre(risultato, info_ricerca, tipo="presenza"):
    try:
        fig, ax = plt.subplots(figsize=(10, 6)); ruota_verifica = info_ricerca.get('ruota_verifica', 'N/D'); numeri_spia_str = ", ".join(info_ricerca.get('numeri_spia', ['N/D'])); ruote_analisi_str = ", ".join(info_ricerca.get('ruote_analisi', [])); res_estratti = risultato.get('estratto', {})
        if tipo == "presenza": dati = res_estratti.get('presenza', {}).get('top', pd.Series(dtype='float64')).to_dict(); percentuali = res_estratti.get('presenza', {}).get('percentuali', pd.Series(dtype='float64')).to_dict(); base_conteggio = risultato.get('totale_trigger', 0); titolo = f"Presenza ESTRATTI su {ruota_verifica} (dopo Spia {numeri_spia_str} su {ruote_analisi_str})"; ylabel = f"N. Serie Trigger ({base_conteggio} totali)"
        else: dati = res_estratti.get('frequenza', {}).get('top', pd.Series(dtype='float64')).to_dict(); percentuali = res_estratti.get('frequenza', {}).get('percentuali', pd.Series(dtype='float64')).to_dict(); base_conteggio = sum(res_estratti.get('frequenza', {}).get('top', pd.Series(dtype='float64')).values) if dati else 0; titolo = f"Frequenza ESTRATTI su {ruota_verifica} (dopo Spia {numeri_spia_str} su {ruote_analisi_str})"; ylabel = "N. Occorrenze Totali (Estratti)"
        numeri = list(dati.keys()); valori = list(dati.values()); perc = [percentuali.get(num, 0.0) for num in numeri]
        if not numeri: ax.text(0.5, 0.5, "Nessun dato", ha='center'); ax.set_title(titolo); ax.axis('off'); plt.close(fig); return None
        bars = ax.bar(numeri, valori, color='skyblue', width=0.6)
        for i, (bar, p) in enumerate(zip(bars, perc)): h = bar.get_height(); p_txt = f'{p:.1f}%' if p > 0.1 else ''; ax.text(bar.get_x() + bar.get_width()/2., h + 0.1, p_txt, ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_xlabel('Numeri Estratti su ' + ruota_verifica); ax.set_ylabel(ylabel); ax.set_title(titolo, fontsize=12); ax.set_ylim(0, max(valori or [1]) * 1.15); ax.yaxis.grid(True, linestyle='--', alpha=0.7); ax.tick_params(axis='x', rotation=45)
        info_text = f"Ruote An: {ruote_analisi_str} | Spia: {numeri_spia_str} | Trigger: {risultato.get('totale_trigger', 0)}"; fig.text(0.5, 0.01, info_text, ha='center', fontsize=9); plt.tight_layout(pad=3.0); return fig
    except Exception as e: print(f"Errore crea_grafico_barre: {e}"); traceback.print_exc(); plt.close(fig); return None # Assicura chiusura figura in caso di errore

def crea_tabella_lotto(risultato, info_ricerca, tipo="presenza"):
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        ruota_verifica = info_ricerca.get('ruota_verifica', 'N/D')
        numeri_spia_str = ", ".join(info_ricerca.get('numeri_spia', ['N/D']))
        ruote_analisi_str = ", ".join(info_ricerca.get('ruote_analisi', []))
        n_trigger = risultato.get('totale_trigger', 0)
        numeri_lotto = np.arange(1, 91).reshape(9, 10)
        res_estratti = risultato.get('estratto', {})
        
        # Recupera il dizionario delle percentuali
        if tipo == "presenza":
            percentuali_serie = res_estratti.get('all_percentuali_presenza', pd.Series(dtype='float64'))
            titolo = f"Tabella Lotto - Presenza ESTRATTI su {ruota_verifica} (dopo Spia {numeri_spia_str} su {ruote_analisi_str})"
        else:
            percentuali_serie = res_estratti.get('all_percentuali_frequenza', pd.Series(dtype='float64'))
            titolo = f"Tabella Lotto - Frequenza ESTRATTI su {ruota_verifica} (dopo Spia {numeri_spia_str} su {ruote_analisi_str})"
        
        # Crea una copia del dizionario per sicurezza e per debugging
        percentuali = percentuali_serie.to_dict() if not percentuali_serie.empty else {}
        
        # Debug: stampa alcuni valori per verificare la struttura
        print(f"DEBUG: primi 5 elementi in percentuali: {list(percentuali.items())[:5]}")
        print(f"DEBUG: Numero totale di percentuali disponibili: {len(percentuali)}")
        
        # Normalizzazione per la colorazione
        colors_norm = np.full(numeri_lotto.shape, 0.9)
        valid_perc = [p for p in percentuali.values() if pd.notna(p) and p > 0]
        max_perc = max(valid_perc) if valid_perc else 1
        if max_perc == 0: 
            max_perc = 1
        
        print(f"DEBUG: max_perc = {max_perc}")
        
        # Calcola i valori di normalizzazione
        found_values = 0
        for i in range(9):
            for j in range(10):
                num = numeri_lotto[i, j]
                # Prova entrambi i formati: con e senza zero iniziale
                num_str = str(num).zfill(2)
                num_str_alt = str(num)
                
                # Cerca il valore con entrambi i formati
                perc_val = percentuali.get(num_str)
                if perc_val is None:
                    perc_val = percentuali.get(num_str_alt)
                
                if perc_val is not None and pd.notna(perc_val) and perc_val > 0:
                    colors_norm[i, j] = 0.9 - (0.9 * perc_val / max_perc)
                    found_values += 1
        
        print(f"DEBUG: Trovati {found_values} numeri con percentuali > 0")
        
        # Crea una griglia per maggiore chiarezza
        for i in range(10):
            ax.axvline(i - 0.5, color='gray', linestyle='-', alpha=0.3)
        for i in range(10):
            ax.axhline(i - 0.5, color='gray', linestyle='-', alpha=0.3)
        
        # Disegna le celle e i testi
        for i in range(9):
            for j in range(10):
                num = numeri_lotto[i, j]
                norm_color = colors_norm[i, j]
                cell_color = "white"
                text_color = "black"
                
                # Calcola il colore in base all'intensità (più chiaro->più scuro)
                if norm_color < 0.9:
                    intensity = (0.9 - norm_color) / 0.9
                    r = int(220 * (1 - intensity))
                    g = int(230 * (1 - intensity))
                    b = int(255 * (1 - intensity/2))
                    cell_color = f"#{r:02x}{g:02x}{b:02x}"
                    
                    if intensity > 0.6:
                        text_color = "white"
                
                # Evidenzia con un bordo più spesso
                edge_color = 'gray'
                if norm_color < 0.9:
                    edge_color = 'black'
                
                # Crea un rettangolo di sfondo per ogni cella
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color=cell_color, 
                                    alpha=1.0, edgecolor=edge_color, linewidth=1)
                ax.add_patch(rect)
                
                # Aggiungi il numero al centro
                ax.text(j, i, num, ha="center", va="center", color=text_color, 
                       fontsize=10, fontweight="bold")
        
        # Imposta i limiti degli assi e rimuovi i tick
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(8.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Aggiungi titolo e informazioni
        plt.title(titolo, fontsize=14, pad=15)
        info_text = f"Ruote An: {ruote_analisi_str} | Spia: {numeri_spia_str} | Trigger: {n_trigger}"
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=9)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        return fig
    
    except Exception as e:
        print(f"Errore crea_tabella_lotto: {e}")
        traceback.print_exc()
        if 'fig' in locals():
            plt.close(fig)
        return None  # Assicura chiusura figura in caso di errore
def crea_heatmap_correlazione(risultati, info_ricerca, tipo="presenza"):
    fig = None # <<--- CORREZIONE 1: Inizializza fig a None prima del try
    try:
        # --- Inizio: Controlli preliminari e inizializzazioni ---
        if len(risultati) < 2:
            print("Heatmap richiede almeno 2 risultati validi.")
            return None
        numeri_spia_str = ", ".join(info_ricerca.get('numeri_spia', ['N/D']))
        ruote_analisi_str = ", ".join(info_ricerca.get('ruote_analisi', []))
        ruote_verifica_list = [r[0] for r in risultati] # Lista dei nomi delle ruote
        all_numeri_estratti = set()

        # --- CORREZIONE 2: Assicurati che l'inizializzazione sia qui ---
        percentuali_per_ruota = {} # Dizionario che conterrà {nome_ruota: {num: perc, ...}}

        # --- Ciclo 1: Estrai le percentuali per ogni ruota valida ---
        print(f"Heatmap '{tipo}': Elaborazione {len(risultati)} risultati...") # Debug
        for ruota_v, spia_str_ignored, res in risultati:
            if res is None or not isinstance(res, dict):
                 print(f"Attenzione [Heatmap]: risultato non valido per {ruota_v}, skippato.")
                 continue

            # Ottieni il dizionario delle percentuali per il tipo richiesto (presenza/frequenza)
            res_tipo = res.get('estratto', {}) # Cerca sempre sotto 'estratto' per ora
            perc_dict = res_tipo.get(f'all_percentuali_{tipo}', pd.Series(dtype='float64')).to_dict()

            # Se non ci sono percentuali per questa ruota, salta (ma non causa errore)
            if not perc_dict:
                print(f"Attenzione [Heatmap]: Nessuna percentuale '{tipo}' trovata per {ruota_v}.")
                continue

            # Aggiungi la ruota e le sue percentuali al dizionario principale
            # Questa riga (ex linea 72) ora modifica il dizionario GIA' inizializzato
            percentuali_per_ruota[ruota_v] = {}
            for num, perc in perc_dict.items():
                if pd.notna(perc) and perc > 0:
                    percentuali_per_ruota[ruota_v][num] = perc
                    all_numeri_estratti.add(num)
            print(f"  [Heatmap] Elaborata {ruota_v}: {len(percentuali_per_ruota[ruota_v])} numeri con perc > 0") # Debug

        # --- Controlli post-ciclo ---
        if not percentuali_per_ruota: # Se nessuna ruota aveva dati validi
             print("Errore [Heatmap]: Nessuna ruota con dati validi trovata dopo il ciclo.")
             return None
        if not all_numeri_estratti:
            print("Errore [Heatmap]: Nessun numero estratto con percentuale > 0 trovato in totale.")
            return None

        # --- Prepara la matrice per la heatmap ---
        all_numeri_sorted = sorted(list(all_numeri_estratti), key=int)
        # Usa solo le ruote che hanno effettivamente dati in percentuali_per_ruota
        ruote_verifica_usate_heatmap = sorted(list(percentuali_per_ruota.keys()))
        n_ruote_effettive = len(ruote_verifica_usate_heatmap)
        n_numeri = len(all_numeri_sorted)

        if n_ruote_effettive < 2: # Ricontrolla dopo aver filtrato
             print(f"Heatmap richiede almeno 2 ruote CON DATI validi. Trovate: {n_ruote_effettive}")
             return None

        matrice = np.full((n_ruote_effettive, n_numeri), np.nan) # Crea matrice vuota

        # --- Ciclo 2: Popola la matrice ---
        for i, ruota_v in enumerate(ruote_verifica_usate_heatmap):
             # Non serve più 'if ruota_v in percentuali_per_ruota:' perché abbiamo già filtrato
             for j, num in enumerate(all_numeri_sorted):
                 # Prendi la percentuale dalla sotto-mappa, default a NaN se il numero non c'era per quella ruota
                 matrice[i, j] = percentuali_per_ruota[ruota_v].get(num, np.nan)

        # --- Crea la figura e gli assi (SOLO ORA) ---
        fig, ax = plt.subplots(figsize=(min(18, n_numeri * 0.5 + 2), max(4, n_ruote_effettive * 0.4 + 1)))
        max_val = np.nanmax(matrice) if not np.all(np.isnan(matrice)) else 1
        if max_val == 0: max_val = 1

        # --- Disegna la heatmap ---
        if SEABORN_AVAILABLE:
            sns.heatmap(matrice, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=all_numeri_sorted, yticklabels=ruote_verifica_usate_heatmap, ax=ax, linewidths=.5, linecolor='gray', cbar=True, vmin=0, vmax=max_val, annot_kws={"size": 7})
            ax.tick_params(axis='x', rotation=90, labelsize=8); ax.tick_params(axis='y', rotation=0, labelsize=8)
        else: # Fallback Matplotlib
            cmap = plt.get_cmap("YlGnBu"); cmap.set_bad(color='lightgray')
            im = ax.imshow(matrice, cmap=cmap, vmin=0, vmax=max_val, aspect='auto')
            ax.set_xticks(np.arange(n_numeri)); ax.set_yticks(np.arange(n_ruote_effettive))
            ax.set_xticklabels(all_numeri_sorted, rotation=90, fontsize=7); ax.set_yticklabels(ruote_verifica_usate_heatmap, fontsize=7);
            for i in range(n_ruote_effettive):
                for j in range(n_numeri):
                    val = matrice[i, j]
                    if not np.isnan(val):
                        text_color = "black" if val < 0.7 * max_val else "white"
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=6)
            cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
            cbar.ax.set_ylabel(f"% {tipo.capitalize()}", rotation=-90, va="bottom", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        # --- Titoli e layout finali ---
        titolo = f"Heatmap {tipo.capitalize()} ESTRATTI: Ruote Verifica vs Numeri\n(Spia {numeri_spia_str} su {ruote_analisi_str})"
        plt.title(titolo, fontsize=11, pad=15); plt.xlabel("Numeri Estratti", fontsize=9); plt.ylabel("Ruote di Verifica", fontsize=9);
        plt.tight_layout()
        return fig # Ritorna la figura creata con successo

    except Exception as e:
        print(f"Errore crea_heatmap_correlazione: {e}")
        traceback.print_exc()
        # --- CORREZIONE 1 (continua): Chiudi fig solo se esiste ---
        if fig:
            plt.close(fig)
        return None # Ritorna None in caso di errore

def visualizza_grafici(risultati_da_visualizzare, info_globale_ricerca, n_estrazioni_usate):
    try:
        if not risultati_da_visualizzare:
            messagebox.showinfo("Nessun Risultato", "Nessun risultato valido da visualizzare.")
            return

        win = tk.Toplevel()
        win.title("Visualizzazione Grafica Risultati Incrociati (Estratti)")
        win.geometry("1300x850")
        win.minsize(900, 600)
        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def create_scrollable_tab(parent, tab_name):
            tab = ttk.Frame(parent)
            parent.add(tab, text=tab_name)
            canvas = tk.Canvas(tab)
            scrollbar_y = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
            scrollbar_x = ttk.Scrollbar(tab, orient="horizontal", command=canvas.xview)
            scrollable_frame = ttk.Frame(canvas)
            scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
            canvas.pack(side="top", fill="both", expand=True)
            scrollbar_y.pack(side="right", fill="y")
            scrollbar_x.pack(side="bottom", fill="x")
            return scrollable_frame

        barre_frame = create_scrollable_tab(notebook, "Grafici a Barre (Estratti)")
        tabelle_frame = create_scrollable_tab(notebook, "Tabelle Lotto (Estratti)")
        heatmap_frame = create_scrollable_tab(notebook, "Heatmap Incrociata (Estratti)")

        # --- Sezione Grafici a Barre ---
        ttk.Label(barre_frame, text="Grafici a Barre per Ruota di Verifica", style="Header.TLabel").pack(pady=10)
        for ruota_v, spia_str, risultato in risultati_da_visualizzare:
             if risultato:
                 info_specifica = info_globale_ricerca.copy()
                 info_specifica['ruota_verifica'] = ruota_v
                 ruota_bar_frame = ttk.LabelFrame(barre_frame, text=f"Ruota Verifica: {ruota_v}")
                 ruota_bar_frame.pack(fill="x", expand=False, padx=10, pady=10)
                 fig_presenza = crea_grafico_barre(risultato, info_specifica, "presenza")
                 fig_frequenza = crea_grafico_barre(risultato, info_specifica, "frequenza")
                 if fig_presenza:
                     canvas_p = FigureCanvasTkAgg(fig_presenza, master=ruota_bar_frame)
                     canvas_p.draw()
                     canvas_p.get_tk_widget().pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                 if fig_frequenza:
                     canvas_f = FigureCanvasTkAgg(fig_frequenza, master=ruota_bar_frame)
                     canvas_f.draw()
                     canvas_f.get_tk_widget().pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
                 if not fig_presenza and not fig_frequenza:
                     ttk.Label(ruota_bar_frame, text="Nessun grafico generato per questa ruota").pack(padx=5, pady=5)

        # --- Sezione Tabelle Lotto ---
        ttk.Label(tabelle_frame, text="Tabelle Lotto per Ruota di Verifica", style="Header.TLabel").pack(pady=10)
        for ruota_v, spia_str, risultato in risultati_da_visualizzare:
            if risultato:
                info_specifica = info_globale_ricerca.copy()
                info_specifica['ruota_verifica'] = ruota_v
                ruota_tab_frame = ttk.LabelFrame(tabelle_frame, text=f"Ruota Verifica: {ruota_v}")
                ruota_tab_frame.pack(fill="x", expand=False, padx=10, pady=10)
                fig_tab_presenza = crea_tabella_lotto(risultato, info_specifica, "presenza")
                fig_tab_frequenza = crea_tabella_lotto(risultato, info_specifica, "frequenza")
                if fig_tab_presenza:
                    canvas_tp = FigureCanvasTkAgg(fig_tab_presenza, master=ruota_tab_frame)
                    canvas_tp.draw()
                    canvas_tp.get_tk_widget().pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                if fig_tab_frequenza:
                    canvas_tf = FigureCanvasTkAgg(fig_tab_frequenza, master=ruota_tab_frame)
                    canvas_tf.draw()
                    canvas_tf.get_tk_widget().pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
                if not fig_tab_presenza and not fig_tab_frequenza:
                    ttk.Label(ruota_tab_frame, text="Nessuna tabella generata per questa ruota").pack(padx=5, pady=5)

        # --- Sezione Heatmap ---
        ttk.Label(heatmap_frame, text="Heatmap Incrociata tra Ruote di Verifica", style="Header.TLabel").pack(pady=10)
        risultati_validi_heatmap = [r for r in risultati_da_visualizzare if r[2] is not None and isinstance(r[2], dict)]
        if len(risultati_validi_heatmap) < 2:
            ttk.Label(heatmap_frame, text="Richiede >= 2 Ruote Verifica con risultati validi.").pack(padx=20, pady=20)
        else:
            heatmap_p_fig = crea_heatmap_correlazione(risultati_validi_heatmap, info_globale_ricerca, "presenza")
            heatmap_f_fig = crea_heatmap_correlazione(risultati_validi_heatmap, info_globale_ricerca, "frequenza")
            if heatmap_p_fig:
                ttk.Label(heatmap_frame, text="--- Heatmap Presenza ---", font=("Helvetica", 11, "bold")).pack(pady=(10,5))
                canvas_hp = FigureCanvasTkAgg(heatmap_p_fig, master=heatmap_frame)
                canvas_hp.draw()
                canvas_hp.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            else:
                ttk.Label(heatmap_frame, text="Nessuna Heatmap Presenza generata (dati insufficienti/invalidi).").pack(pady=10)
            if heatmap_f_fig:
                ttk.Label(heatmap_frame, text="--- Heatmap Frequenza ---", font=("Helvetica", 11, "bold")).pack(pady=(10,5))
                canvas_hf = FigureCanvasTkAgg(heatmap_f_fig, master=heatmap_frame)
                canvas_hf.draw()
                canvas_hf.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            else:
                ttk.Label(heatmap_frame, text="Nessuna Heatmap Frequenza generata (dati insufficienti/invalidi).").pack(pady=10)

    except Exception as e:
        messagebox.showerror("Errore Visualizzazione", f"Errore durante la creazione della finestra grafici:\n{e}")
        traceback.print_exc()
    finally:
        plt.close('all')


# =============================================================================
# FUNZIONI DI LOGICA
# =============================================================================
def carica_dati(file_path, start_date=None, end_date=None):
    try:
        if not os.path.exists(file_path): print(f"Errore File: {file_path} non esiste."); return None
        with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        dates, ruote, numeri = [], [], []; seen_rows = set(); fmt_ok = '%Y/%m/%d'

        # print(f"Leggendo file {os.path.basename(file_path)}: {len(lines)} righe") # Meno verboso
        filtered_count = 0

        for line in lines:
            line = line.strip();
            if not line: continue
            parts = line.split();
            if len(parts) < 7: continue
            data_str, ruota_str, nums_orig = parts[0], parts[1].upper(), parts[2:7]

            try: data_dt_val = datetime.datetime.strptime(data_str, fmt_ok); [int(n) for n in nums_orig]
            except ValueError: continue

            if start_date and end_date:
                try:
                    # Confronta solo le date, non l'ora
                    if data_dt_val.date() < start_date.date() or data_dt_val.date() > end_date.date():
                        filtered_count += 1
                        continue
                except Exception:
                    continue

            key = f"{data_str}_{ruota_str}"
            if key in seen_rows:
                continue
            seen_rows.add(key)

            dates.append(data_str); ruote.append(ruota_str); numeri.append(nums_orig)

        # print(f"File {os.path.basename(file_path)}: {len(dates)} righe utilizzabili, {filtered_count} filtrate per data")

        if not dates: # print(f"{os.path.basename(file_path)} vuoto o invalido dopo filtri.");
            return None

        df = pd.DataFrame({'Data': dates, 'Ruota': ruote, 'Numero1': [n[0] for n in numeri], 'Numero2': [n[1] for n in numeri],'Numero3': [n[2] for n in numeri], 'Numero4': [n[3] for n in numeri],'Numero5': [n[4] for n in numeri]})
        df['Data'] = pd.to_datetime(df['Data'], format=fmt_ok)

        for col in ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']:
             df[col] = df[col].apply(lambda x: str(int(x)).zfill(2) if pd.notna(x) and str(x).isdigit() and 1 <= int(x) <= 90 else pd.NA)

        df.dropna(subset=['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5'], inplace=True)
        df = df.sort_values(by='Data').reset_index(drop=True)

        if df.empty:
            # print(f"DataFrame finale vuoto dopo pulizia/dropna per {os.path.basename(file_path)}")
            return None

        # print(f"DataFrame finale per {os.path.basename(file_path)}: {df.shape[0]} righe, date da {df['Data'].min().date()} a {df['Data'].max().date()}")
        return df

    except FileNotFoundError:
         print(f"!!! ERRORE CRITICO: File non trovato {file_path}")
         return None
    except Exception as e:
        print(f"!!! Errore grave lettura/elaborazione file {os.path.basename(file_path)}: {e}"); traceback.print_exc(); return None

def analizza_ruota_verifica(df_verifica, date_trigger_sorted, n_estrazioni, nome_ruota_verifica):
    # print(f"Analisi E/A/T {nome_ruota_verifica} ({len(date_trigger_sorted)} trigger).") # Meno verboso
    if df_verifica is None or df_verifica.empty: return None, "Df verifica vuoto."
    df_verifica = df_verifica.sort_values(by='Data').drop_duplicates(subset=['Data']).reset_index(drop=True)
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']
    n_trigger = len(date_trigger_sorted); date_series_verifica = df_verifica['Data']
    freq_estratti = {}; freq_ambi = {}; freq_terne = {}; pres_estratti = {}; pres_ambi = {}; pres_terne = {}
    for data_t in date_trigger_sorted:
        try:
            start_index = date_series_verifica.searchsorted(data_t, side='right')
            if start_index >= len(date_series_verifica):
                 continue
        except Exception as e_search:
            # print(f"Errore searchsorted per data {data_t} su {nome_ruota_verifica}: {e_search}") # Meno verboso
            continue
        df_successive = df_verifica.iloc[start_index : start_index + n_estrazioni]
        estratti_unici_finestra = set(); ambi_unici_finestra = set(); terne_unici_finestra = set()
        if not df_successive.empty:
            for _, row in df_successive.iterrows():
                numeri_estratti = [row[col] for col in colonne_numeri if pd.notna(row[col])]
                if not numeri_estratti: continue
                numeri_estratti.sort()
                for num in numeri_estratti:
                    freq_estratti[num] = freq_estratti.get(num, 0) + 1
                    estratti_unici_finestra.add(num)
                if len(numeri_estratti) >= 2:
                    for ambo in itertools.combinations(numeri_estratti, 2):
                        ambo_key = tuple(ambo)
                        freq_ambi[ambo_key] = freq_ambi.get(ambo_key, 0) + 1
                        ambi_unici_finestra.add(ambo_key)
                if len(numeri_estratti) >= 3:
                    for terno in itertools.combinations(numeri_estratti, 3):
                        terno_key = tuple(terno)
                        freq_terne[terno_key] = freq_terne.get(terno_key, 0) + 1
                        terne_unici_finestra.add(terno_key)
        for num in estratti_unici_finestra: pres_estratti[num] = pres_estratti.get(num, 0) + 1
        for ambo in ambi_unici_finestra: pres_ambi[ambo] = pres_ambi.get(ambo, 0) + 1
        for terno in terne_unici_finestra: pres_terne[terno] = pres_terne.get(terno, 0) + 1

    results = {'totale_trigger': n_trigger}
    for tipo, freq_dict, pres_dict in [('estratto', freq_estratti, pres_estratti), ('ambo', freq_ambi, pres_ambi), ('terno', freq_terne, pres_terne)]:
        if not freq_dict:
            results[tipo] = None
            continue
        freq_series = pd.Series(freq_dict, dtype=int).sort_index()
        pres_series = pd.Series(pres_dict, dtype=int)
        pres_series = pres_series.reindex(freq_series.index, fill_value=0).sort_index()
        total_freq = freq_series.sum();
        perc_freq = (freq_series / total_freq * 100).round(2) if total_freq > 0 else pd.Series(0.0, index=freq_series.index);
        perc_pres = (pres_series / n_trigger * 100).round(2) if n_trigger > 0 else pd.Series(0.0, index=pres_series.index)
        top_pres = pres_series.sort_values(ascending=False).head(10);
        top_freq = freq_series.sort_values(ascending=False).head(10)
        freq_dei_top_pres = freq_series.reindex(top_pres.index).fillna(0).astype(int);
        perc_freq_dei_top_pres = perc_freq.reindex(top_pres.index).fillna(0.0);
        pres_dei_top_freq = pres_series.reindex(top_freq.index).fillna(0).astype(int);
        perc_pres_dei_top_freq = perc_pres.reindex(top_freq.index).fillna(0.0)
        results[tipo] = {
            'presenza': {'top': top_pres,'percentuali': perc_pres.reindex(top_pres.index).fillna(0.0),'frequenze': freq_dei_top_pres,'perc_frequenza': perc_freq_dei_top_pres},
            'frequenza': {'top': top_freq,'percentuali': perc_freq.reindex(top_freq.index).fillna(0.0),'presenze': pres_dei_top_freq,'perc_presenza': perc_pres_dei_top_freq},
            'all_percentuali_presenza': perc_pres,'all_percentuali_frequenza': perc_freq,'full_presenze': pres_series,'full_frequenze': freq_series}

    # print(f"Analisi E/A/T {nome_ruota_verifica} OK.") # Meno verboso
    if results.get('estratto') or results.get('ambo') or results.get('terno'):
        return results, None
    else:
        return None, f"Nessun risultato (E/A/T) trovato su {nome_ruota_verifica}."

def analizza_antecedenti(df_ruota, numeri_obiettivo, n_precedenti, nome_ruota):
    # print(f"\nAnalisi Antecedenti per {numeri_obiettivo} su {nome_ruota} ({n_precedenti} prec.)"); # Meno verboso
    if df_ruota is None or df_ruota.empty: return None, "DataFrame vuoto.";
    if not numeri_obiettivo: return None, "Nessun obiettivo specificato.";
    if n_precedenti <= 0: return None, "N. precedenti deve essere > 0.";
    # print(f"DataFrame shape: {df_ruota.shape}")
    if not df_ruota.empty:
        # print(f"Range date: {df_ruota['Data'].min().date()} - {df_ruota['Data'].max().date()}")
        pass
    else:
        # print("DataFrame vuoto, impossibile determinare range date.")
        return None, "DataFrame vuoto passato ad analizza_antecedenti."

    df_ruota = df_ruota.sort_values(by='Data').reset_index(drop=True);
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']
    mask_obiettivo = df_ruota[colonne_numeri].isin(numeri_obiettivo).any(axis=1);
    indices_obiettivo = df_ruota.index[mask_obiettivo].tolist();
    n_occorrenze_obiettivo = len(indices_obiettivo)
    # print(f"Trovate {n_occorrenze_obiettivo} occorrenze (estrazioni) con obiettivi {numeri_obiettivo} su {nome_ruota} NEL PERIODO DATO.");
    if n_occorrenze_obiettivo == 0: return None, f"Obiettivi {numeri_obiettivo} non trovati su {nome_ruota} nel periodo selezionato.";

    frequenze_antecedenti = {}; presenze_antecedenti = {}; actual_base_presenza = 0
    for idx_obiettivo in indices_obiettivo:
        if idx_obiettivo < n_precedenti:
            continue
        actual_base_presenza += 1
        start_idx_prec = idx_obiettivo - n_precedenti
        end_idx_prec = idx_obiettivo
        df_precedenti = df_ruota.iloc[start_idx_prec:end_idx_prec]
        if not df_precedenti.empty:
            numeri_nella_finestra = df_precedenti[colonne_numeri].values.flatten();
            numeri_unici_finestra = set()
            for num in numeri_nella_finestra:
                if pd.notna(num) and num is not None:
                    frequenze_antecedenti[num] = frequenze_antecedenti.get(num, 0) + 1;
                    numeri_unici_finestra.add(num)
            for num_unico in numeri_unici_finestra:
                presenze_antecedenti[num_unico] = presenze_antecedenti.get(num_unico, 0) + 1

    if actual_base_presenza == 0:
         # print(f"Nessuna finestra antecedente valida trovata per {nome_ruota} con n_precedenti={n_precedenti}")
         empty_stats = {'top': pd.Series(dtype=int), 'percentuali': pd.Series(dtype=float), 'frequenze': pd.Series(dtype=int), 'perc_frequenza': pd.Series(dtype=float)}
         empty_freq_stats = {'top': pd.Series(dtype=int), 'percentuali': pd.Series(dtype=float), 'presenze': pd.Series(dtype=int), 'perc_presenza': pd.Series(dtype=float)}
         return {'totale_occorrenze_obiettivo': n_occorrenze_obiettivo, 'base_presenza_antecedenti': 0, 'numeri_obiettivo': numeri_obiettivo, 'n_precedenti': n_precedenti, 'nome_ruota': nome_ruota, 'presenza': empty_stats, 'frequenza': empty_freq_stats}, f"Nessuna finestra antecedente valida trovata."

    if not frequenze_antecedenti:
        empty_stats = {'top': pd.Series(dtype=int), 'percentuali': pd.Series(dtype=float), 'frequenze': pd.Series(dtype=int), 'perc_frequenza': pd.Series(dtype=float)}
        empty_freq_stats = {'top': pd.Series(dtype=int), 'percentuali': pd.Series(dtype=float), 'presenze': pd.Series(dtype=int), 'perc_presenza': pd.Series(dtype=float)}
        return {'totale_occorrenze_obiettivo': n_occorrenze_obiettivo, 'base_presenza_antecedenti': actual_base_presenza, 'numeri_obiettivo': numeri_obiettivo, 'n_precedenti': n_precedenti, 'nome_ruota': nome_ruota, 'presenza': empty_stats, 'frequenza': empty_freq_stats}, f"Nessun numero antecedente trovato nelle {actual_base_presenza} finestre valide."

    antecedenti_freq_series = pd.Series(frequenze_antecedenti, dtype=int).sort_index();
    antecedenti_pres_series = pd.Series(presenze_antecedenti, dtype=int).reindex(antecedenti_freq_series.index, fill_value=0).sort_index()
    total_antecedenti_freq = antecedenti_freq_series.sum(); base_presenza = actual_base_presenza
    perc_antecedenti_freq = pd.Series(0.0, index=antecedenti_freq_series.index)
    if total_antecedenti_freq > 0: perc_antecedenti_freq = (antecedenti_freq_series / total_antecedenti_freq * 100).round(2)
    perc_antecedenti_pres = pd.Series(0.0, index=antecedenti_pres_series.index)
    if base_presenza > 0: perc_antecedenti_pres = (antecedenti_pres_series / base_presenza * 100).round(2)
    top_antecedenti_pres = antecedenti_pres_series.sort_values(ascending=False).head(10);
    top_antecedenti_freq = antecedenti_freq_series.sort_values(ascending=False).head(10)
    freq_dei_top_pres = antecedenti_freq_series.reindex(top_antecedenti_pres.index).fillna(0).astype(int);
    perc_freq_dei_top_pres = perc_antecedenti_freq.reindex(top_antecedenti_pres.index).fillna(0.0);
    pres_dei_top_freq = antecedenti_pres_series.reindex(top_antecedenti_freq.index).fillna(0).astype(int);
    perc_pres_dei_top_freq = perc_antecedenti_pres.reindex(top_antecedenti_freq.index).fillna(0.0)
    # print(f"Analisi antecedenti {nome_ruota} completata (occorrenze: {n_occorrenze_obiettivo}, finestre valide: {base_presenza})"); # Meno verboso
    return {'presenza': {'top': top_antecedenti_pres,'percentuali': perc_antecedenti_pres.reindex(top_antecedenti_pres.index).fillna(0.0),'frequenze': freq_dei_top_pres,'perc_frequenza': perc_freq_dei_top_pres},
            'frequenza': {'top': top_antecedenti_freq,'percentuali': perc_antecedenti_freq.reindex(top_antecedenti_freq.index).fillna(0.0),'presenze': pres_dei_top_freq,'perc_presenza': perc_pres_dei_top_freq},
            'totale_occorrenze_obiettivo': n_occorrenze_obiettivo,'base_presenza_antecedenti': base_presenza,'numeri_obiettivo': numeri_obiettivo,'n_precedenti': n_precedenti,'nome_ruota': nome_ruota}, None

def aggiorna_risultati_globali(risultati_nuovi, info_ricerca=None, modalita="successivi"):
    global risultati_globali, info_ricerca_globale, button_verifica_esiti, button_visualizza # Assicura che i bottoni siano accessibili
    button_verifica_esiti.config(state=tk.DISABLED); button_visualizza.config(state=tk.DISABLED)
    if modalita == "successivi":
        risultati_globali = risultati_nuovi if risultati_nuovi is not None else []
        info_ricerca_globale = info_ricerca if info_ricerca is not None else {}
        has_valid_results = bool(risultati_globali) and any(res[2] is not None for res in risultati_globali if len(res) > 2)
        if has_valid_results and info_ricerca_globale.get('top_combinati') and info_ricerca_globale.get('date_trigger_ordinate'):
            button_visualizza.config(state=tk.NORMAL);
            button_verifica_esiti.config(state=tk.NORMAL)
        # else: # Debug:
        #      print("Debug: Pulsanti non abilitati. Risultati validi:", has_valid_results, "| Top combinati:", bool(info_ricerca_globale.get('top_combinati')), "| Date trigger:", bool(info_ricerca_globale.get('date_trigger_ordinate')))
    else:
        risultati_globali = []
        info_ricerca_globale = {}

def salva_risultati():
    global risultato_text # Assicura accesso
    risultato_text.config(state=tk.NORMAL); results_content = risultato_text.get(1.0, tk.END).strip(); risultato_text.config(state=tk.DISABLED)
    default_msgs = ["Benvenuto", "Ricerca in corso...", "Nessun risultato", "Seleziona Ruota"]
    is_empty = not results_content or any(msg in results_content for msg in default_msgs)
    if is_empty: messagebox.showinfo("Nessun Risultato", "Niente da salvare."); return
    f_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")], title="Salva Risultati")
    if f_path:
        try:
            with open(f_path, "w", encoding="utf-8") as f: f.write(results_content)
            messagebox.showinfo("Salvataggio OK", f"Salvati in:\n{f_path}")
        except Exception as e: messagebox.showerror("Errore Salvataggio", f"Errore:\n{e}")

def format_ambo_terno(combinazione):
    return "-".join(map(str, combinazione))

# --- Funzione cerca_numeri ---
def cerca_numeri(modalita="successivi"):
    global risultati_globali, info_ricerca_globale, risultato_text, root, start_date_entry, end_date_entry # Assicura accesso
    global listbox_ruote_analisi, listbox_ruote_verifica, entry_numeri_spia, estrazioni_entry_succ # Per successivi
    global listbox_ruote_analisi_ant, entry_numeri_obiettivo, estrazioni_entry_ant # Per antecedenti

    if not mappa_file_ruote():
        messagebox.showerror("Errore File", "Impossibile mappare i file delle ruote. Verifica la cartella selezionata.")
        return
    if not file_ruote:
         messagebox.showerror("Errore File", "Nessun file di estrazione valido trovato nella cartella selezionata.")
         return

    risultati_globali = []
    info_ricerca_globale = {}

    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
    risultato_text.insert(tk.END, f"Ricerca {modalita} in corso...\n"); risultato_text.see(tk.END); root.update_idletasks()
    aggiorna_risultati_globali([], {}, modalita=modalita)

    try:
        start_dt = start_date_entry.get_date(); end_dt = end_date_entry.get_date();
        if start_dt > end_dt: raise ValueError("Data inizio successiva a data fine.")
        start_ts = pd.Timestamp(start_dt); end_ts = pd.Timestamp(end_dt)
    except ValueError as e: messagebox.showerror("Input Invalido", f"Date non valide: {e}"); risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END); risultato_text.insert(tk.END, "Errore input date."); risultato_text.config(state=tk.DISABLED); return
    except Exception as e: messagebox.showerror("Errore Date", f"Errore lettura date: {e}"); risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END); risultato_text.insert(tk.END, "Errore input date."); risultato_text.config(state=tk.DISABLED); return

    messaggi_output = []; risultati_per_grafici_local = []

    if modalita == "successivi":
        ruote_analisi_indices = listbox_ruote_analisi.curselection(); ruote_verifica_indices = listbox_ruote_verifica.curselection()
        if not ruote_analisi_indices: messagebox.showwarning("Manca Input", "Seleziona Ruota/e Analisi (1)."); return
        if not ruote_verifica_indices: messagebox.showwarning("Manca Input", "Seleziona Ruota/e Verifica (3)."); return
        nomi_ruote_analisi = [listbox_ruote_analisi.get(i) for i in ruote_analisi_indices]; nomi_ruote_verifica = [listbox_ruote_verifica.get(i) for i in ruote_verifica_indices]
        numeri_spia_input = set()
        for entry in entry_numeri_spia:
            val = entry.get().strip();
            if val:
                 try: num_int = int(val); assert 1 <= num_int <= 90; numeri_spia_input.add(str(num_int).zfill(2))
                 except (ValueError, AssertionError): messagebox.showwarning("Input Invalido", f"Numero Spia '{val}' non valido (1-90)."); return
        numeri_spia_validi = sorted(list(numeri_spia_input))
        if not numeri_spia_validi: messagebox.showwarning("Manca Input", "Inserisci Numero/i Spia valido/i (2)."); return
        try: n_estrazioni = int(estrazioni_entry_succ.get()); assert 1 <= n_estrazioni <= 18
        except (ValueError, AssertionError): messagebox.showerror("Input Invalido", "N. Estrazioni Successive (4) deve essere 1-18."); return

        info_ricerca_corrente = {'numeri_spia': numeri_spia_validi, 'ruote_analisi': nomi_ruote_analisi, 'ruote_verifica': nomi_ruote_verifica, 'n_estrazioni': n_estrazioni, 'start_date': start_ts, 'end_date': end_ts}
        all_date_trigger = set(); messaggi_output.append("--- FASE 1: Ricerca Date Uscita Spia ---"); # print("\n--- FASE 1: Ricerca Date Trigger ---")
        colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']
        for nome_ra in nomi_ruote_analisi:
            fp = file_ruote.get(nome_ra);
            if not fp: msg = f"[{nome_ra}] File non trovato nella mappa."; messaggi_output.append(msg); continue
            # print(f"Carico An: {nome_ra} con date {start_ts.date()} - {end_ts.date()}...");
            df_an = carica_dati(fp, start_ts, end_ts)
            if df_an is None or df_an.empty: msg = f"[{nome_ra}] Nessun dato An nel periodo specificato."; messaggi_output.append(msg); continue
            mask = df_an[colonne_numeri].isin(numeri_spia_validi).any(axis=1)
            dates_found_series = df_an.loc[mask, 'Data']
            if not dates_found_series.empty:
                 dates_found = pd.to_datetime(dates_found_series.unique())
                 all_date_trigger.update(dates_found)
                 msg = f"[{nome_ra}] Trovate {len(dates_found)} date trigger."; messaggi_output.append(msg)
            else:
                 messaggi_output.append(f"[{nome_ra}] Nessuna uscita spia {', '.join(numeri_spia_validi)} trovata.")

        if not all_date_trigger:
            msg = f"\nNESSUNA USCITA TROVATA per Spia {', '.join(numeri_spia_validi)} su Ruote Analisi nel periodo selezionato.";
            risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END); risultato_text.insert(tk.END, "\n".join(messaggi_output) + msg); risultato_text.config(state=tk.DISABLED);
            aggiorna_risultati_globali([], {}, modalita=modalita)
            return

        date_trigger_ordinate = sorted(list(all_date_trigger)); n_trigger_totali = len(date_trigger_ordinate)
        msg_f1_ok = f"\nFASE 1 OK: {n_trigger_totali} date trigger totali trovate (dal {date_trigger_ordinate[0].date()} al {date_trigger_ordinate[-1].date()})."; messaggi_output.append(msg_f1_ok)
        info_ricerca_corrente['date_trigger_ordinate'] = date_trigger_ordinate

        messaggi_output.append("\n--- FASE 2: Analisi Ruote Verifica ---"); # print("\n--- FASE 2: Analisi Ruote Verifica ---")
        risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END); risultato_text.insert(tk.END, "\n".join(messaggi_output) + "\nAnalisi ruote verifica..."); risultato_text.see(tk.END); root.update_idletasks()
        num_ruote_verifica_ok = 0; df_cache_verifica = {}
        for nome_rv in nomi_ruote_verifica:
            if nome_rv not in df_cache_verifica:
                fp_v = file_ruote.get(nome_rv);
                if not fp_v: msg = f"[{nome_rv}] File verifica non trovato nella mappa."; messaggi_output.append(msg); continue
                # print(f"Carico Ver: {nome_rv} con date {start_ts.date()} - {end_ts.date()}...");
                df_ver_full = carica_dati(fp_v, start_ts, end_ts)
                if df_ver_full is None or df_ver_full.empty:
                     msg = f"[{nome_rv}] Nessun dato Ver nel periodo specificato."; messaggi_output.append(msg);
                     df_cache_verifica[nome_rv] = None
                     continue
                df_cache_verifica[nome_rv] = df_ver_full.copy()
            else:
                # print(f"Uso cache per Ver: {nome_rv}")
                df_ver_full = df_cache_verifica[nome_rv]
                if df_ver_full is None: continue

            res_ver, err_ver = analizza_ruota_verifica(df_ver_full, date_trigger_ordinate, n_estrazioni, nome_rv)
            if err_ver: msg = f"[{nome_rv}] Errore analisi: {err_ver}"; messaggi_output.append(msg)
            elif res_ver:
                risultati_per_grafici_local.append((nome_rv, ", ".join(numeri_spia_validi), res_ver))
                num_ruote_verifica_ok += 1
                msg_res_v = f"\n=== Risultati Verifica: {nome_rv} ===\n(Base: {res_ver['totale_trigger']} trigger | Spia: {', '.join(numeri_spia_validi)} | Succ: {n_estrazioni})"
                res_estratti = res_ver.get('estratto')
                if res_estratti:
                    msg_res_v += f"\n--- Estratti ---\n  Top Presenza (% su {res_ver['totale_trigger']} trigger):\n"
                    if not res_estratti['presenza']['top'].empty:
                        for i, (num, pres) in enumerate(res_estratti['presenza']['top'].items()): perc_p = res_estratti['presenza']['percentuali'].get(num, 0.0); freq_p = res_estratti['presenza']['frequenze'].get(num, 0); msg_res_v += f"    {i+1}. {num}: {pres} ({perc_p:.1f}%) [Freq: {freq_p}]\n"
                    else: msg_res_v += "    Nessuno.\n"
                    msg_res_v += f"  Top Frequenza (N. Volte):\n"
                    if not res_estratti['frequenza']['top'].empty:
                         for i, (num, freq) in enumerate(res_estratti['frequenza']['top'].items()): pres_f = res_estratti['frequenza']['presenze'].get(num, 0); msg_res_v += f"    {i+1}. {num}: {freq} volte [Pres: {pres_f}]\n"
                    else: msg_res_v += "    Nessuno.\n"
                else: msg_res_v += "\n--- Estratti: Nessun risultato ---\n"
                res_ambi = res_ver.get('ambo');
                if res_ambi:
                    msg_res_v += f"\n--- Ambi ---\n  Top Presenza (% su {res_ver['totale_trigger']} trigger):\n";
                    if not res_ambi['presenza']['top'].empty:
                        for i, (ambo, pres) in enumerate(res_ambi['presenza']['top'].items()): ambo_str = format_ambo_terno(ambo); perc_p = res_ambi['presenza']['percentuali'].get(ambo, 0.0); freq = res_ambi['presenza']['frequenze'].get(ambo, 0); msg_res_v += f"    {i+1}. {ambo_str}: {pres} ({perc_p:.1f}%) [Freq: {freq}]\n"
                    else: msg_res_v += "    Nessuno.\n"
                    msg_res_v += f"  Top Frequenza (N. Volte):\n"
                    if not res_ambi['frequenza']['top'].empty:
                         for i, (ambo, freq) in enumerate(res_ambi['frequenza']['top'].items()): ambo_str = format_ambo_terno(ambo); pres = res_ambi['frequenza']['presenze'].get(ambo, 0); msg_res_v += f"    {i+1}. {ambo_str}: {freq} volte [Pres: {pres}]\n"
                    else: msg_res_v += "    Nessuno.\n"
                else: msg_res_v += "\n--- Ambi: Nessun risultato ---\n"
                res_terne = res_ver.get('terno');
                if res_terne:
                    msg_res_v += f"\n--- Terne ---\n  Top Presenza (% su {res_ver['totale_trigger']} trigger):\n";
                    if not res_terne['presenza']['top'].empty:
                        for i, (terno, pres) in enumerate(res_terne['presenza']['top'].items()): terno_str = format_ambo_terno(terno); perc_p = res_terne['presenza']['percentuali'].get(terno, 0.0); freq = res_terne['presenza']['frequenze'].get(terno, 0); msg_res_v += f"    {i+1}. {terno_str}: {pres} ({perc_p:.1f}%) [Freq: {freq}]\n"
                    else: msg_res_v += "    Nessuno.\n"
                    msg_res_v += f"  Top Frequenza (N. Volte):\n"
                    if not res_terne['frequenza']['top'].empty:
                         for i, (terno, freq) in enumerate(res_terne['frequenza']['top'].items()): terno_str = format_ambo_terno(terno); pres = res_terne['frequenza']['presenze'].get(terno, 0); msg_res_v += f"    {i+1}. {terno_str}: {freq} volte [Pres: {pres}]\n"
                    else: msg_res_v += "    Nessuno.\n"
                else: msg_res_v += "\n--- Terne: Nessun risultato ---\n"
                messaggi_output.append(msg_res_v)
            else:
                 msg = f"[{nome_rv}] Nessun risultato E/A/T trovato nelle estrazioni successive."
                 messaggi_output.append(msg)
            messaggi_output.append("-" * 40)

        if risultati_per_grafici_local and num_ruote_verifica_ok > 0:
            # print("\n--- FASE 3: Aggregazione Risultati Combinati ---") # Meno verboso
            messaggi_output.append("\n\n=== RISULTATI COMBINATI (Tutte le Ruote Verifica Analizzate) ===")
            ruote_v_con_risultati = [r[0] for r in risultati_per_grafici_local if r[2] is not None]
            ruote_v_str = ', '.join(ruote_v_con_risultati)
            messaggi_output.append(f"\nStatistiche aggregate basate sulle ruote con risultati: {ruote_v_str}")
            messaggi_output.append(f"(Base: {n_trigger_totali} trigger da {', '.join(nomi_ruote_analisi)} | Spia: {', '.join(numeri_spia_validi)} | Succ: {n_estrazioni})\n")
            top_combinati_per_verifica = {'estratto': [], 'ambo': [], 'terno': []}
            peso_presenza = 0.6; peso_frequenza = 0.4
            for tipo in ['estratto', 'ambo', 'terno']:
                messaggi_output.append(f"\n--- Combinati: {tipo.upper()} ---")
                combined_pres_dict = {}; combined_freq_dict = {}; has_data_for_type = False
                for _, _, res in risultati_per_grafici_local:
                    if res is None: continue
                    res_tipo = res.get(tipo)
                    if res_tipo:
                        pres_series = res_tipo.get('full_presenze', pd.Series(dtype=int));
                        freq_series = res_tipo.get('full_frequenze', pd.Series(dtype=int))
                        if not pres_series.empty:
                             has_data_for_type = True;
                             for item, count in pres_series.items(): combined_pres_dict[item] = combined_pres_dict.get(item, 0) + count
                        if not freq_series.empty:
                             has_data_for_type = True;
                             for item, count in freq_series.items(): combined_freq_dict[item] = combined_freq_dict.get(item, 0) + count
                if not has_data_for_type:
                     messaggi_output.append(f"    Nessun risultato combinato per {tipo}.\n"); continue
                combined_pres = pd.Series(combined_pres_dict, dtype=int)
                combined_freq = pd.Series(combined_freq_dict, dtype=int)
                all_items_index = combined_pres.index.union(combined_freq.index)
                combined_pres = combined_pres.reindex(all_items_index, fill_value=0).sort_index()
                combined_freq = combined_freq.reindex(all_items_index, fill_value=0).sort_index()
                total_presence_opportunities = n_trigger_totali * num_ruote_verifica_ok
                comb_perc_presenza = pd.Series(0.0, index=combined_pres.index);
                if total_presence_opportunities > 0: comb_perc_presenza = (combined_pres / total_presence_opportunities * 100)
                max_freq = combined_freq.max()
                comb_freq_normalizzata = pd.Series(0.0, index=combined_freq.index)
                if max_freq > 0: comb_freq_normalizzata = (combined_freq / max_freq * 100)
                punteggio_combinato = (peso_presenza * comb_perc_presenza) + (peso_frequenza * comb_freq_normalizzata)
                punteggio_combinato = punteggio_combinato.round(2).sort_values(ascending=False)
                top_combinati_punteggio = punteggio_combinato.head(10)
                if not top_combinati_punteggio.empty:
                    top_combinati_per_verifica[tipo] = top_combinati_punteggio.head(5).index.tolist()
                messaggi_output.append(f"  Top 10 Combinati per Punteggio (Pres {peso_presenza*100:.0f}%, Freq {peso_frequenza*100:.0f}%):\n")
                if not top_combinati_punteggio.empty:
                    for i, (item, score) in enumerate(top_combinati_punteggio.items()):
                        item_str = format_ambo_terno(item) if isinstance(item, tuple) else item
                        pres_comb = combined_pres.get(item, 0); freq_comb = combined_freq.get(item, 0);
                        perc_p_comb_avg = comb_perc_presenza.get(item, 0.0)
                        messaggi_output.append(f"    {i+1}. {item_str}: Punt={score:.2f} (PresAvg: {perc_p_comb_avg:.1f}%, FreqTot: {freq_comb})\n")
                else:
                    messaggi_output.append("    Nessuno.\n")
            messaggi_output.append("-" * 40)
            info_ricerca_corrente['top_combinati'] = top_combinati_per_verifica
        elif num_ruote_verifica_ok == 0:
            messaggi_output.append("\nNessuna Ruota di Verifica valida o con risultati trovati. Impossibile calcolare risultati combinati.")

        aggiorna_risultati_globali(risultati_per_grafici_local, info_ricerca_corrente, modalita="successivi")

    elif modalita == "antecedenti":
        ruote_analisi_ant_indices = listbox_ruote_analisi_ant.curselection();
        if not ruote_analisi_ant_indices: messagebox.showwarning("Manca Input", "Seleziona Ruota/e da Analizzare (1)."); return
        nomi_ruote_analisi_ant = [listbox_ruote_analisi_ant.get(i) for i in ruote_analisi_ant_indices];
        numeri_obiettivo_input = set()
        for entry in entry_numeri_obiettivo:
            val = entry.get().strip();
            if val:
                try: num_int = int(val); assert 1 <= num_int <= 90; numeri_obiettivo_input.add(str(num_int).zfill(2))
                except (ValueError, AssertionError): messagebox.showwarning("Input Invalido", f"Numero Obiettivo '{val}' non valido (1-90)."); return
        numeri_obiettivo_validi = sorted(list(numeri_obiettivo_input));
        if not numeri_obiettivo_validi: messagebox.showwarning("Manca Input", "Inserisci Numero/i Obiettivo valido/i (2)."); return
        try:
            n_precedenti = int(estrazioni_entry_ant.get());
            if n_precedenti > 18: print("Attenzione: N. precedenti > 18, potrebbe rallentare.")
            assert n_precedenti >= 1
        except (ValueError, AssertionError): messagebox.showerror("Input Invalido", "N. Estrazioni Precedenti (3) deve essere >= 1 (consigliato <= 18)."); return

        messaggi_output.append(f"--- Analisi Antecedenti (Marker) ---"); # print("\n--- Esecuzione Analisi Antecedenti ---")
        df_cache_antecedenti = {}
        for nome_ruota_ant in nomi_ruote_analisi_ant:
            df_ruota_corrente = None
            if nome_ruota_ant not in df_cache_antecedenti:
                 fp_ant = file_ruote.get(nome_ruota_ant);
                 if not fp_ant: msg = f"[{nome_ruota_ant}] File non trovato nella mappa."; messaggi_output.append(msg); continue
                 # print(f"Carico dati per {nome_ruota_ant} (Antecedenti) date {start_ts.date()} - {end_ts.date()}...");
                 df_ant_full = carica_dati(fp_ant, start_ts, end_ts)
                 df_cache_antecedenti[nome_ruota_ant] = df_ant_full
            else:
                 # print(f"Uso cache per Ant: {nome_ruota_ant}")
                 df_ant_full = df_cache_antecedenti[nome_ruota_ant]

            if df_ant_full is None or df_ant_full.empty:
                msg = f"[{nome_ruota_ant}] Nessun dato per questa ruota nel periodo specificato."
                messaggi_output.append(msg); continue

            if 'Ruota' in df_ant_full.columns:
                # print(f"    Shape PRIMA del filtro per RUOTA {nome_ruota_ant}: {df_ant_full.shape}")
                ruota_map = {
                    'BARI': ['BARI', 'BA'], 'CAGLIARI': ['CAGLIARI', 'CA'], 'FIRENZE': ['FIRENZE', 'FI'],
                    'GENOVA': ['GENOVA', 'GE'], 'MILANO': ['MILANO', 'MI'], 'NAPOLI': ['NAPOLI', 'NA'],
                    'PALERMO': ['PALERMO', 'PA'], 'ROMA': ['ROMA', 'RM'], 'TORINO': ['TORINO', 'TO'],
                    'VENEZIA': ['VENEZIA', 'VE'], 'NAZIONALE': ['NAZIONALE', 'NAZ']
                }
                nomi_validi_ruota = ruota_map.get(nome_ruota_ant, [nome_ruota_ant])
                maschera_ruota = df_ant_full['Ruota'].isin(nomi_validi_ruota)
                df_ruota_corrente = df_ant_full[maschera_ruota].copy()
                # print(f"    Shape DOPO il filtro per RUOTA {nomi_validi_ruota}: {df_ruota_corrente.shape}")
                if df_ruota_corrente.empty:
                    # print(f"    Valori unici nella colonna Ruota del file caricato: {df_ant_full['Ruota'].unique()}")
                    msg = f"[{nome_ruota_ant}] Nessun dato trovato per la ruota specifica '{nome_ruota_ant}' nel file caricato (periodo {start_ts.date()}-{end_ts.date()})."
                    messaggi_output.append(msg); continue
            else:
                df_ruota_corrente = df_ant_full.copy()
                # print(f"    Nessuna colonna 'Ruota' trovata, assumo file specifico per {nome_ruota_ant}. Shape: {df_ruota_corrente.shape}")

            # print(f"    Chiamata ad analizza_antecedenti per {nome_ruota_ant} con {df_ruota_corrente.shape[0]} righe.")
            res_ant, err_ant = analizza_antecedenti(df_ruota_corrente, numeri_obiettivo_validi, n_precedenti, nome_ruota_ant)

            if err_ant: msg = f"[{nome_ruota_ant}] Errore analisi: {err_ant}"; messaggi_output.append(msg)
            elif res_ant:
                if res_ant.get('base_presenza_antecedenti', 0) > 0 and (not res_ant['presenza']['top'].empty or not res_ant['frequenza']['top'].empty):
                    msg_res_ant = f"\n=== Risultati Antecedenti (Marker) per: {nome_ruota_ant} ===\n(Obiettivi: {', '.join(res_ant['numeri_obiettivo'])} | Prec: {res_ant['n_precedenti']} estr. | Occ. Ob.: {res_ant['totale_occorrenze_obiettivo']} | Base Pres. Ant.: {res_ant['base_presenza_antecedenti']})"
                    msg_res_ant += f"\n  Top Antecedenti per Presenza (% su {res_ant['base_presenza_antecedenti']} finestre valide):\n"
                    if not res_ant['presenza']['top'].empty:
                        for i, (num, pres) in enumerate(res_ant['presenza']['top'].head(10).items()): perc_p = res_ant['presenza']['percentuali'].get(num, 0.0); freq = res_ant['presenza']['frequenze'].get(num, 0); msg_res_ant += f"    {i+1}. {num}: {pres} ({perc_p:.1f}%) [Freq: {freq}]\n"
                    else: msg_res_ant += "    Nessuno.\n"
                    msg_res_ant += f"  Top Antecedenti per Frequenza (N. volte apparsi prima):\n"
                    if not res_ant['frequenza']['top'].empty:
                         for i, (num, freq) in enumerate(res_ant['frequenza']['top'].head(10).items()): pres = res_ant['frequenza']['presenze'].get(num, 0); msg_res_ant += f"    {i+1}. {num}: {freq} volte [Pres: {pres}]\n"
                    else: msg_res_ant += "    Nessuno.\n"
                    messaggi_output.append(msg_res_ant)
                else:
                    msg = f"[{nome_ruota_ant}] Analisi completata, ma nessun dato antecedente trovato (Occ. Ob.: {res_ant['totale_occorrenze_obiettivo']}, Finestre Valide: {res_ant['base_presenza_antecedenti']})."
                    messaggi_output.append(msg)
            messaggi_output.append("-" * 40)
        # --- FINE CICLO FOR RUOTE ANTECEDENTI ---

        # Resetta risultati globali dopo analisi antecedenti (corretta indentazione)
        aggiorna_risultati_globali([], {}, modalita="antecedenti") # <-- LINEA 961 CORRETTA
        # --- FINE BLOCCO ANTECEDENTI ---

    else: # Questo else appartiene a if modalita == "successivi"
        messagebox.showerror("Errore Interno", f"Modalità sconosciuta: {modalita}")
        risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END); risultato_text.config(state=tk.DISABLED); return

    # --- Output Finale (comune, ora correttamente indentato dopo if/elif/else) ---
    risultato_text.config(state=tk.NORMAL);
    try:
        risultato_text.delete(1.0, tk.END);
        final_output = "\n".join(messaggi_output)
        if not final_output.strip():
            final_output = "Nessun risultato specifico da mostrare."
            if modalita == "antecedenti":
                 final_output += "\n(Possibile causa: nessun dato nel periodo, obiettivi non trovati, o nessuna finestra antecedente valida)."
            elif modalita == "successivi":
                 final_output += "\n(Possibile causa: nessun dato nel periodo, numeri spia non trovati, o nessun risultato nelle estrazioni successive)."

        risultato_text.insert(tk.END, final_output)
        risultato_text.config(state=tk.DISABLED);
        risultato_text.see("1.0")
    except Exception as e_out:
         print(f"!!! ERRORE durante scrittura output finale: {e_out}")
         traceback.print_exc()
         try:
             risultato_text.delete(1.0, tk.END);
             risultato_text.insert(tk.END, f"ERRORE CRITICO DURANTE OUTPUT:\nImpossibile visualizzare i risultati.\n\nDettagli:\n{e_out}\n\n{traceback.format_exc()}")
             risultato_text.config(state=tk.DISABLED);
         except: pass
# --- FINE FUNZIONE cerca_numeri ---


# =============================================================================
# FUNZIONI PER VERIFICA ESITI
# =============================================================================
def verifica_esiti_combinati(date_triggers, top_combinati, nomi_ruote_verifica, n_verifiche, start_ts, end_ts):
    # print(f"\n--- Verifica Esiti ({n_verifiche} estr. verifica) ---") # Meno verboso
    if not date_triggers: return "Errore: Date trigger non disponibili."
    if not top_combinati: return "Errore: Top combinati non disponibili."
    if not nomi_ruote_verifica: return "Errore: Ruote di verifica non specificate."
    top_estratti = top_combinati.get('estratto', []); top_ambi = top_combinati.get('ambo', []); top_terne = top_combinati.get('terno', [])
    vincite = {'estratto': {e: 0 for e in top_estratti}, 'ambo': {a: 0 for a in top_ambi}, 'terno': {t: 0 for t in top_terne}}
    set_top_estratti = set(top_estratti); set_top_ambi = set(top_ambi); set_top_terne = set(top_terne)
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']
    df_cache_ver = {}
    # print("Pre-caricamento dati ruote verifica per esiti...");
    ruote_valide_per_verifica = []
    for nome_rv in nomi_ruote_verifica:
        fp_v = file_ruote.get(nome_rv)
        if not fp_v:
            # print(f"Attenzione: File per {nome_rv} non trovato per verifica esiti.")
            continue
        # print(f"Carico {nome_rv} per verifica ({start_ts.date()} - {end_ts.date()})")
        df_ver = carica_dati(fp_v, start_ts, end_ts)
        if df_ver is not None and not df_ver.empty:
            df_cache_ver[nome_rv] = df_ver.sort_values(by='Data').drop_duplicates(subset=['Data']).reset_index(drop=True)
            ruote_valide_per_verifica.append(nome_rv)
        # else:
             # print(f"Attenzione: Impossibile caricare dati per verifica ruota {nome_rv}")

    if not ruote_valide_per_verifica:
         return "Errore: Nessuna ruota di verifica valida trovata o caricata per la verifica esiti."
    # print(f"Pre-caricamento completato per {len(ruote_valide_per_verifica)} ruote.")
    casi_totali_effettivi = len(date_triggers) * len(ruote_valide_per_verifica)
    if casi_totali_effettivi == 0: return "Nessun caso da verificare (0 trigger o 0 ruote valide)."
    # print(f"Verifica effettiva su {casi_totali_effettivi} casi (trigger * ruote valide)")
    # progress_step = max(1, len(date_triggers) // 20); # Rimuovi stampa progresso
    for trigger_idx, data_t in enumerate(date_triggers):
        # if (trigger_idx + 1) % progress_step == 0: print(f"Verifica trigger {trigger_idx + 1}/{len(date_triggers)}...") # Rimuovi stampa progresso
        for nome_rv in ruote_valide_per_verifica:
            df_verifica = df_cache_ver[nome_rv]; date_series_verifica = df_verifica['Data']
            try:
                start_index = date_series_verifica.searchsorted(data_t, side='right')
                if start_index >= len(date_series_verifica): continue
            except Exception as e_search_v:
                # print(f"Errore searchsorted verifica per {data_t} su {nome_rv}: {e_search_v}") # Meno verboso
                continue
            df_finestra_verifica = df_verifica.iloc[start_index : start_index + n_verifiche]
            if not df_finestra_verifica.empty:
                found_in_this_case_window = {'estratto': set(), 'ambo': set(), 'terno': set()}
                for _, row in df_finestra_verifica.iterrows():
                    numeri_draw = [row[col] for col in colonne_numeri if pd.notna(row[col])]
                    if not numeri_draw: continue
                    numeri_draw.sort(); set_numeri_draw = set(numeri_draw)
                    if top_estratti:
                        intersezione_e = set_numeri_draw.intersection(set_top_estratti)
                        for num_vinto in intersezione_e:
                            if num_vinto not in found_in_this_case_window['estratto']:
                                vincite['estratto'][num_vinto] += 1; found_in_this_case_window['estratto'].add(num_vinto)
                    if top_ambi and len(numeri_draw) >= 2:
                        ambi_draw = set(itertools.combinations(numeri_draw, 2))
                        intersezione_a = ambi_draw.intersection(set_top_ambi)
                        for ambo_vinto in intersezione_a:
                            if ambo_vinto not in found_in_this_case_window['ambo']:
                                vincite['ambo'][ambo_vinto] += 1; found_in_this_case_window['ambo'].add(ambo_vinto)
                    if top_terne and len(numeri_draw) >= 3:
                        terne_draw = set(itertools.combinations(numeri_draw, 3))
                        intersezione_t = terne_draw.intersection(set_top_terne)
                        for terno_vinto in intersezione_t:
                            if terno_vinto not in found_in_this_case_window['terno']:
                                vincite['terno'][terno_vinto] += 1; found_in_this_case_window['terno'].add(terno_vinto)

    output_verifica = [f"\n\n=== VERIFICA ESITI ({n_verifiche} Colpi) ==="]
    output_verifica.append(f"Verifica eseguita su {casi_totali_effettivi} casi ({len(date_triggers)} trigger * {len(ruote_valide_per_verifica)} ruote valide).")
    for tipo in ['estratto', 'ambo', 'terno']:
        output_verifica.append(f"\n--- Esiti {tipo.upper()} ---")
        top_items = top_combinati.get(tipo, []); vincite_tipo = vincite.get(tipo, {})
        if not top_items: output_verifica.append("    Nessun Top da verificare per questo tipo."); continue
        if not vincite_tipo: output_verifica.append("    Errore interno: Contatori vincite non trovati."); continue
        for item in top_items:
            item_str = format_ambo_terno(item) if isinstance(item, tuple) else item
            n_vincite = vincite_tipo.get(item, 0)
            perc_vincita = round(n_vincite / casi_totali_effettivi * 100, 2) if casi_totali_effettivi > 0 else 0.0
            output_verifica.append(f"    - {item_str}: {n_vincite} vincite ({perc_vincita}%)")
    # print("Verifica Esiti completata.") # Meno verboso
    return "\n".join(output_verifica)

def esegui_verifica_esiti():
    global info_ricerca_globale, risultato_text, root, estrazioni_entry_verifica # Assicura accesso
    # print("Avvio verifica esiti..."); # Meno verboso
    risultato_text.config(state=tk.NORMAL); risultato_text.insert(tk.END, "\n\nVerifica esiti in corso..."); risultato_text.see(tk.END); root.update_idletasks()
    date_triggers = info_ricerca_globale.get('date_trigger_ordinate'); top_combinati = info_ricerca_globale.get('top_combinati'); nomi_ruote_verifica = info_ricerca_globale.get('ruote_verifica'); start_ts = info_ricerca_globale.get('start_date'); end_ts = info_ricerca_globale.get('end_date')
    if not all([date_triggers is not None, top_combinati, nomi_ruote_verifica, start_ts, end_ts]):
        messagebox.showerror("Errore Verifica", "Informazioni dall'analisi 'Successivi' precedente non trovate o incomplete. Esegui prima 'Cerca Successivi' con successo.");
        risultato_text.insert(tk.END, "\nErrore: Dati precedenti non trovati."); risultato_text.config(state=tk.DISABLED); return
    try: n_verifiche = int(estrazioni_entry_verifica.get()); assert 1 <= n_verifiche <= 18
    except (ValueError, AssertionError): messagebox.showerror("Input Invalido", f"Numero Estrazioni Verifica (1-18) non valido."); risultato_text.insert(tk.END, f"\nErrore: N. Estrazioni Verifica non valido."); risultato_text.config(state=tk.DISABLED); return
    try:
        risultato_stringa = verifica_esiti_combinati(date_triggers, top_combinati, nomi_ruote_verifica, n_verifiche, start_ts, end_ts);
        risultato_text.insert(tk.END, risultato_stringa)
    except Exception as e:
        error_msg = f"\nErrore durante la verifica esiti: {e}"; print(error_msg); traceback.print_exc();
        risultato_text.insert(tk.END, error_msg)
        messagebox.showerror("Errore Verifica Esiti", f"Si è verificato un errore:\n{e}")
    risultato_text.see(tk.END); risultato_text.config(state=tk.DISABLED)


# =============================================================================
# Funzione Wrapper per Visualizza Grafici
# =============================================================================
def visualizza_grafici_successivi():
    global risultati_globali, info_ricerca_globale # Assicura accesso
    has_valid_global_results = bool(risultati_globali) and any(res[2] is not None for res in risultati_globali if len(res) > 2)
    if info_ricerca_globale and 'ruote_verifica' in info_ricerca_globale and has_valid_global_results:
        risultati_validi_da_mostrare = [r for r in risultati_globali if r[2] is not None]
        if risultati_validi_da_mostrare:
             visualizza_grafici(risultati_validi_da_mostrare, info_ricerca_globale, info_ricerca_globale.get('n_estrazioni', 5))
        else:
             messagebox.showinfo("Grafici non Disponibili", "Nessun risultato valido trovato nell'analisi 'Successivi' per generare grafici.")
    else:
         messagebox.showinfo("Grafici non Disponibili", "Esegui prima 'Cerca Successivi' e assicurati che produca risultati validi.")


# =============================================================================
# GUI e Mainloop
# =============================================================================
# (Codice GUI invariato dalla versione precedente - l'indentazione qui è corretta)
root = tk.Tk(); root.title("Analisi Lotto v3.8 - FIX"); root.geometry("1150x800"); root.minsize(1000, 650); root.configure(bg="#f0f0f0")
style = ttk.Style(); style.theme_use('clam'); style.configure("TFrame", background="#f0f0f0"); style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 10)); style.configure("TButton", font=("Segoe UI", 10), padding=5); style.configure("Title.TLabel", font=("Segoe UI", 11, "bold")); style.configure("Header.TLabel", font=("Segoe UI", 12, "bold")); style.configure("Small.TLabel", background="#f0f0f0", font=("Segoe UI", 8)); style.configure("TEntry", padding=3); style.configure("TListbox", font=("Consolas", 10)); style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), background="#f0f0f0"); style.configure("TNotebook.Tab", padding=[10, 5], font=("Segoe UI", 10))
main_frame = ttk.Frame(root, padding=10); main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5); cartella_frame = ttk.Frame(main_frame); cartella_frame.pack(fill=tk.X, pady=(0, 10)); ttk.Label(cartella_frame, text="Cartella Estrazioni:", style="Title.TLabel").pack(side=tk.LEFT, padx=(0, 5)); cartella_entry = ttk.Entry(cartella_frame, width=60); cartella_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5); btn_sfoglia = ttk.Button(cartella_frame, text="Sfoglia..."); btn_sfoglia.pack(side=tk.LEFT, padx=5)
notebook = ttk.Notebook(main_frame, style="TNotebook"); notebook.pack(fill=tk.X, pady=10)
tab_successivi = ttk.Frame(notebook, padding=10); notebook.add(tab_successivi, text=' Analisi Numeri Successivi (E/A/T) '); controls_frame_succ = ttk.Frame(tab_successivi); controls_frame_succ.pack(fill=tk.X); controls_frame_succ.columnconfigure(0, weight=1); controls_frame_succ.columnconfigure(1, weight=1); controls_frame_succ.columnconfigure(2, weight=0); controls_frame_succ.columnconfigure(3, weight=0)
ruote_analisi_outer_frame = ttk.Frame(controls_frame_succ); ruote_analisi_outer_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5)); ttk.Label(ruote_analisi_outer_frame, text="1. Ruote Analisi:", style="Title.TLabel").pack(anchor="w"); ttk.Label(ruote_analisi_outer_frame, text="(CTRL/SHIFT per multipla)", style="Small.TLabel").pack(anchor="w", pady=(0, 5)); ruote_analisi_list_frame = ttk.Frame(ruote_analisi_outer_frame); ruote_analisi_list_frame.pack(fill=tk.BOTH, expand=True); scrollbar_ruote_analisi = ttk.Scrollbar(ruote_analisi_list_frame); scrollbar_ruote_analisi.pack(side=tk.RIGHT, fill=tk.Y); listbox_ruote_analisi = tk.Listbox(ruote_analisi_list_frame, height=10, selectmode=tk.EXTENDED, exportselection=False, font=("Consolas", 10), selectbackground="#005A9E", selectforeground="white", yscrollcommand=scrollbar_ruote_analisi.set); listbox_ruote_analisi.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar_ruote_analisi.config(command=listbox_ruote_analisi.yview)
ruote_verifica_outer_frame = ttk.Frame(controls_frame_succ); ruote_verifica_outer_frame.grid(row=0, column=1, sticky="nsew", padx=5); ttk.Label(ruote_verifica_outer_frame, text="3. Ruote Verifica:", style="Title.TLabel").pack(anchor="w"); ttk.Label(ruote_verifica_outer_frame, text="(CTRL/SHIFT per multipla)", style="Small.TLabel").pack(anchor="w", pady=(0, 5)); ruote_verifica_list_frame = ttk.Frame(ruote_verifica_outer_frame); ruote_verifica_list_frame.pack(fill=tk.BOTH, expand=True); scrollbar_ruote_verifica = ttk.Scrollbar(ruote_verifica_list_frame); scrollbar_ruote_verifica.pack(side=tk.RIGHT, fill=tk.Y); listbox_ruote_verifica = tk.Listbox(ruote_verifica_list_frame, height=10, selectmode=tk.EXTENDED, exportselection=False, font=("Consolas", 10), selectbackground="#005A9E", selectforeground="white", yscrollcommand=scrollbar_ruote_verifica.set); listbox_ruote_verifica.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar_ruote_verifica.config(command=listbox_ruote_verifica.yview)
center_controls_frame_succ = ttk.Frame(controls_frame_succ); center_controls_frame_succ.grid(row=0, column=2, sticky="ns", padx=5); spia_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 2. Numeri Spia (1-90) ", padding=5); spia_frame_succ.pack(fill=tk.X, pady=(0, 5)); spia_entry_container_succ = ttk.Frame(spia_frame_succ); spia_entry_container_succ.pack(fill=tk.X, pady=5); entry_numeri_spia = [];
for i in range(5): entry = ttk.Entry(spia_entry_container_succ, width=5, justify=tk.CENTER, font=("Segoe UI", 10)); entry.pack(side=tk.LEFT, padx=3, ipady=2); entry_numeri_spia.append(entry)
estrazioni_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 4. Estrazioni Successive ", padding=5); estrazioni_frame_succ.pack(fill=tk.X, pady=5); ttk.Label(estrazioni_frame_succ, text="Quante (1-18):", style="Small.TLabel").pack(anchor="w"); estrazioni_entry_succ = ttk.Entry(estrazioni_frame_succ, width=5, justify=tk.CENTER, font=("Segoe UI", 10)); estrazioni_entry_succ.pack(anchor="w", pady=2, ipady=2); estrazioni_entry_succ.insert(0, "5")
verifica_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 5. Verifica Esiti ", padding=5); verifica_frame_succ.pack(fill=tk.X, pady=5); ttk.Label(verifica_frame_succ, text="Estrazioni Verifica (1-18):", style="Small.TLabel").pack(anchor="w"); estrazioni_entry_verifica = ttk.Entry(verifica_frame_succ, width=5, justify=tk.CENTER, font=("Segoe UI", 10)); estrazioni_entry_verifica.pack(anchor="w", pady=2, ipady=2); estrazioni_entry_verifica.insert(0, "9")
buttons_frame_succ = ttk.Frame(controls_frame_succ); buttons_frame_succ.grid(row=0, column=3, sticky="ns", padx=(10, 0)); button_cerca_succ = ttk.Button(buttons_frame_succ, text="Cerca Successivi", command=lambda: cerca_numeri(modalita="successivi")); button_cerca_succ.pack(pady=5, fill=tk.X, ipady=3); button_verifica_esiti = ttk.Button(buttons_frame_succ, text="Verifica Esiti", command=esegui_verifica_esiti); button_verifica_esiti.pack(pady=5, fill=tk.X, ipady=3); button_verifica_esiti.config(state=tk.DISABLED)
tab_antecedenti = ttk.Frame(notebook, padding=10); notebook.add(tab_antecedenti, text=' Analisi Numeri Antecedenti (Marker) '); controls_frame_ant = ttk.Frame(tab_antecedenti); controls_frame_ant.pack(fill=tk.X); controls_frame_ant.columnconfigure(0, weight=1); controls_frame_ant.columnconfigure(1, weight=0); controls_frame_ant.columnconfigure(2, weight=0); ruote_analisi_ant_outer_frame = ttk.Frame(controls_frame_ant); ruote_analisi_ant_outer_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10)); ttk.Label(ruote_analisi_ant_outer_frame, text="1. Ruote da Analizzare:", style="Title.TLabel").pack(anchor="w"); ttk.Label(ruote_analisi_ant_outer_frame, text="(Obiettivo e antecedenti cercati qui)", style="Small.TLabel").pack(anchor="w", pady=(0, 5)); ruote_analisi_ant_list_frame = ttk.Frame(ruote_analisi_ant_outer_frame); ruote_analisi_ant_list_frame.pack(fill=tk.BOTH, expand=True); scrollbar_ruote_analisi_ant = ttk.Scrollbar(ruote_analisi_ant_list_frame); scrollbar_ruote_analisi_ant.pack(side=tk.RIGHT, fill=tk.Y); listbox_ruote_analisi_ant = tk.Listbox(ruote_analisi_ant_list_frame, height=10, selectmode=tk.EXTENDED, exportselection=False, font=("Consolas", 10), selectbackground="#005A9E", selectforeground="white", yscrollcommand=scrollbar_ruote_analisi_ant.set); listbox_ruote_analisi_ant.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar_ruote_analisi_ant.config(command=listbox_ruote_analisi_ant.yview); center_controls_frame_ant = ttk.Frame(controls_frame_ant); center_controls_frame_ant.grid(row=0, column=1, sticky="ns", padx=10); obiettivo_frame_ant = ttk.LabelFrame(center_controls_frame_ant, text=" 2. Numeri Obiettivo (1-90) ", padding=5); obiettivo_frame_ant.pack(fill=tk.X, pady=(0, 5)); obiettivo_entry_container_ant = ttk.Frame(obiettivo_frame_ant); obiettivo_entry_container_ant.pack(fill=tk.X, pady=5); entry_numeri_obiettivo = [];
for i in range(5): entry = ttk.Entry(obiettivo_entry_container_ant, width=5, justify=tk.CENTER, font=("Segoe UI", 10)); entry.pack(side=tk.LEFT, padx=3, ipady=2); entry_numeri_obiettivo.append(entry)
estrazioni_frame_ant = ttk.LabelFrame(center_controls_frame_ant, text=" 3. Estrazioni Precedenti ", padding=5); estrazioni_frame_ant.pack(fill=tk.X, pady=5); ttk.Label(estrazioni_frame_ant, text="Quante controllare (>=1):", style="Small.TLabel").pack(anchor="w"); estrazioni_entry_ant = ttk.Entry(estrazioni_frame_ant, width=5, justify=tk.CENTER, font=("Segoe UI", 10)); estrazioni_entry_ant.pack(anchor="w", pady=2, ipady=2); estrazioni_entry_ant.insert(0, "3"); buttons_frame_ant = ttk.Frame(controls_frame_ant); buttons_frame_ant.grid(row=0, column=2, sticky="ns", padx=(10, 0)); button_cerca_ant = ttk.Button(buttons_frame_ant, text="Cerca Antecedenti", command=lambda: cerca_numeri(modalita="antecedenti")); button_cerca_ant.pack(pady=5, fill=tk.X, ipady=3)
common_controls_frame = ttk.Frame(main_frame); common_controls_frame.pack(fill=tk.X, pady=5); dates_frame = ttk.LabelFrame(common_controls_frame, text=" Periodo Analisi (Comune) ", padding=5); dates_frame.pack(side=tk.LEFT, padx=(0,10)); dates_frame.columnconfigure(1, weight=1); ttk.Label(dates_frame, text="Da:", anchor="e").grid(row=0, column=0, padx=2, pady=2, sticky="w"); start_date_default = datetime.date.today() - datetime.timedelta(days=365*3); start_date_entry = DateEntry(dates_frame, width=10, background='#3498db', foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd', font=("Segoe UI", 9), year=start_date_default.year, month=start_date_default.month, day=start_date_default.day); start_date_entry.grid(row=0, column=1, padx=2, pady=2, sticky="ew"); ttk.Label(dates_frame, text="A:", anchor="e").grid(row=1, column=0, padx=2, pady=2, sticky="w"); end_date_entry = DateEntry(dates_frame, width=10, background='#3498db', foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd', font=("Segoe UI", 9)); end_date_entry.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
common_buttons_frame = ttk.Frame(common_controls_frame); common_buttons_frame.pack(side=tk.LEFT, padx=10); button_salva = ttk.Button(common_buttons_frame, text="Salva Risultati", command=salva_risultati); button_salva.pack(side=tk.LEFT, pady=5, padx=5, ipady=3); button_visualizza = ttk.Button(common_buttons_frame, text="Visualizza Grafici\n(Solo Successivi)", command=visualizza_grafici_successivi); button_visualizza.pack(side=tk.LEFT, pady=5, padx=5, ipady=0); button_visualizza.config(state=tk.DISABLED)
ttk.Label(main_frame, text="Risultati Analisi:", style="Header.TLabel").pack(anchor="w", pady=(15, 0)); risultato_outer_frame = ttk.Frame(main_frame); risultato_outer_frame.pack(fill=tk.BOTH, expand=True, pady=5); risultato_scroll_y = ttk.Scrollbar(risultato_outer_frame, orient=tk.VERTICAL); risultato_scroll_y.pack(side=tk.RIGHT, fill=tk.Y); risultato_scroll_x = ttk.Scrollbar(risultato_outer_frame, orient=tk.HORIZONTAL); risultato_scroll_x.pack(side=tk.BOTTOM, fill=tk.X); risultato_text = tk.Text(risultato_outer_frame, wrap=tk.NONE, font=("Consolas", 10), height=15, yscrollcommand=risultato_scroll_y.set, xscrollcommand=risultato_scroll_x.set, state=tk.DISABLED, bd=1, relief="sunken"); risultato_text.pack(fill=tk.BOTH, expand=True); risultato_scroll_y.config(command=risultato_text.yview); risultato_scroll_x.config(command=risultato_text.xview)

def aggiorna_lista_file_gui(target_listbox):
    global file_ruote # Assicura accesso
    target_listbox.config(state=tk.NORMAL) # Abilita prima di modificare
    target_listbox.delete(0, tk.END)
    ruote_ordinate = sorted(file_ruote.keys())
    if ruote_ordinate:
        for r in ruote_ordinate:
            target_listbox.insert(tk.END, r)
    else:
        target_listbox.insert(tk.END, "Nessun file ruota valido")
        target_listbox.config(state=tk.DISABLED)

def mappa_file_ruote():
    global file_ruote, cartella_entry # Assicura accesso
    cartella = cartella_entry.get()
    file_ruote.clear()
    if not cartella or not os.path.isdir(cartella):
        # print(f"Cartella non valida o non specificata: '{cartella}'") # Meno verboso
        return False
    ruote_valide = ['BARI', 'CAGLIARI', 'FIRENZE', 'GENOVA', 'MILANO', 'NAPOLI', 'PALERMO', 'ROMA', 'TORINO', 'VENEZIA', 'NAZIONALE']
    found_files = False
    try:
        # print(f"Scansione cartella: {cartella}") # Meno verboso
        for file in os.listdir(cartella):
            fp = os.path.join(cartella, file)
            if os.path.isfile(fp) and file.lower().endswith(".txt"):
                nome_base_file = os.path.splitext(file)[0].upper()
                if nome_base_file in ruote_valide:
                    file_ruote[nome_base_file] = fp
                    # print(f"  Trovato file valido: {file} -> Ruota: {nome_base_file}") # Meno verboso
                    found_files = True
        # if not found_files: print(f"Nessun file .txt con nome ruota valido trovato in {cartella}") # Meno verboso
        return found_files
    except OSError as e: messagebox.showerror("Errore Lettura Cartella", f"Impossibile leggere la cartella:\n{e}"); return False
    except Exception as e: messagebox.showerror("Errore Inatteso Scansione", f"Errore durante scansione file:\n{e}"); traceback.print_exc(); return False

def on_sfoglia_click():
    global cartella_entry, listbox_ruote_analisi, listbox_ruote_verifica, listbox_ruote_analisi_ant # Assicura accesso
    cartella_sel = filedialog.askdirectory(title="Seleziona Cartella Estrazioni")
    if cartella_sel:
        cartella_entry.delete(0, tk.END)
        cartella_entry.insert(0, cartella_sel)
        if mappa_file_ruote():
            aggiorna_lista_file_gui(listbox_ruote_analisi)
            aggiorna_lista_file_gui(listbox_ruote_verifica)
            aggiorna_lista_file_gui(listbox_ruote_analisi_ant)
        else:
            listbox_ruote_analisi.config(state=tk.NORMAL)
            listbox_ruote_analisi.delete(0, tk.END)
            listbox_ruote_analisi.insert(tk.END, "Nessun file ruota valido")
            listbox_ruote_analisi.config(state=tk.DISABLED)
            listbox_ruote_verifica.config(state=tk.NORMAL)
            listbox_ruote_verifica.delete(0, tk.END)
            listbox_ruote_verifica.insert(tk.END, "Nessun file ruota valido")
            listbox_ruote_verifica.config(state=tk.DISABLED)
            listbox_ruote_analisi_ant.config(state=tk.NORMAL)
            listbox_ruote_analisi_ant.delete(0, tk.END)
            listbox_ruote_analisi_ant.insert(tk.END, "Nessun file ruota valido")
            listbox_ruote_analisi_ant.config(state=tk.DISABLED)
            messagebox.showwarning("Nessun File Trovato", "Nessun file di estrazione .txt valido (es. BARI.txt) trovato nella cartella selezionata.")

btn_sfoglia.config(command=on_sfoglia_click)

def main():
    global root, risultato_text # Assicura accesso
    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0, tk.END)
    risultato_text.insert(tk.END, "Benvenuto!\n\n")
    risultato_text.insert(tk.END, "1. Usa 'Sfoglia...' per selezionare la cartella con i file .txt delle estrazioni.\n")
    risultato_text.insert(tk.END, "2. Seleziona la modalità (Successivi o Antecedenti) e imposta i parametri.\n")
    risultato_text.insert(tk.END, "3. Imposta il periodo di analisi.\n")
    risultato_text.insert(tk.END, "4. Clicca 'Cerca...'.\n")
    risultato_text.insert(tk.END, "5. Dopo 'Cerca Successivi', puoi usare 'Verifica Esiti' e 'Visualizza Grafici'.\n")
    risultato_text.config(state=tk.DISABLED)
    root.mainloop()
    print("\n=========================================")
    print("Finestra Tkinter chiusa. Script terminato.")

if __name__ == "__main__":
    main()