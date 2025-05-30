# -*- coding: utf-8 -*-
# Versione 3.9.7 - ANALISI POSIZIONALE (SPIA + ESITI)
# MODIFICATO PER COPIA/INCOLLA E SALVATAGGIO POPUP

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from tkinter import scrolledtext
import pandas as pd
import numpy as np
import os
from tkcalendar import DateEntry
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
import traceback
import sys
import itertools
from collections import Counter, OrderedDict

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Variabili globali
risultati_globali = []
info_ricerca_globale = {}
file_ruote = {}
MAX_COLPI_GIOCO = 90 # Nuovo limite per i colpi di gioco

# Variabili GUI globali
button_visualizza = None
button_verifica_futura = None
estrazioni_entry_verifica_futura = None
button_verifica_mista = None
text_combinazioni_miste = None
estrazioni_entry_verifica_mista = None

# MODIFICATO/NUOVO per analisi posizionale e GUI
tipo_spia_var_global = None
entry_numeri_spia = []
combo_posizione_spia = None

# Variabili GUI per Numeri Simpatici
entry_numero_target_simpatici = None
listbox_ruote_simpatici = None
entry_top_n_simpatici = None


# =============================================================================
# FUNZIONI GRAFICHE (Come da tua versione 3.9.3 - INVARIATE)
# =============================================================================
def crea_grafico_barre(risultato, info_ricerca, tipo_analisi="estratto", tipo_stat="presenza"):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ruota_verifica = info_ricerca.get('ruota_verifica', 'N/D')
        spia_val = info_ricerca.get('numeri_spia_input', ['N/D'])
        if isinstance(spia_val, list):
            numeri_spia_str = ", ".join(map(str,spia_val))
        elif isinstance(spia_val, tuple):
            numeri_spia_str = "-".join(map(str,spia_val))
        else:
            numeri_spia_str = str(spia_val)
        ruote_analisi_str = ", ".join(info_ricerca.get('ruote_analisi', []))
        res_tipo_analisi = risultato.get(tipo_analisi, {})
        if not res_tipo_analisi:
             ax.text(0.5, 0.5, f"Nessun dato per {tipo_analisi.upper()}", ha='center', va='center')
             ax.set_title(f"{tipo_stat.capitalize()} {tipo_analisi.upper()} su {ruota_verifica}")
             ax.axis('off'); plt.close(fig); return None
        if tipo_stat == "presenza":
            dati_serie = res_tipo_analisi.get('presenza', {}).get('top', pd.Series(dtype='float64'))
            percentuali_serie = res_tipo_analisi.get('presenza', {}).get('percentuali', pd.Series(dtype='float64'))
            base_conteggio = risultato.get('totale_trigger', 0)
            titolo = f"Presenza {tipo_analisi.upper()} su {ruota_verifica}\n(Spia {numeri_spia_str} su {ruote_analisi_str})"
            ylabel = f"N. Serie Trigger ({base_conteggio} totali)"
        else:
            dati_serie = res_tipo_analisi.get('frequenza', {}).get('top', pd.Series(dtype='float64'))
            percentuali_serie = res_tipo_analisi.get('frequenza', {}).get('percentuali', pd.Series(dtype='float64'))
            base_conteggio = sum(dati_serie.values) if not dati_serie.empty else 0
            titolo = f"Frequenza {tipo_analisi.upper()} su {ruota_verifica}\n(Spia {numeri_spia_str} su {ruote_analisi_str})"
            ylabel = f"N. Occorrenze Totali ({tipo_analisi.capitalize()})"
        dati = dati_serie.to_dict(); percentuali = percentuali_serie.to_dict()
        if tipo_analisi in ["ambo", "terno"]:
            numeri = list(dati.keys())
            valori = list(dati.values())
            perc = [percentuali.get(k, 0.0) for k in numeri]
        else:
            numeri = list(dati.keys()); valori = list(dati.values())
            perc = [percentuali.get(num, 0.0) for num in numeri]
        if not numeri:
            ax.text(0.5, 0.5, f"Nessun dato per {tipo_analisi.upper()}", ha='center', va='center')
            ax.set_title(titolo); ax.axis('off'); plt.close(fig); return None
        bars = ax.bar(numeri, valori, color='skyblue', width=0.6)
        for i, (bar, p_val) in enumerate(zip(bars, perc)):
            h = bar.get_height()
            p_txt = f'{p_val:.1f}%' if p_val > 0.01 else ''
            ax.text(bar.get_x() + bar.get_width()/2., h + max(valori or [1])*0.01, p_txt, ha='center', va='bottom', fontweight='bold', fontsize=8)
        ax.set_xlabel(f'{tipo_analisi.capitalize()} su ' + ruota_verifica); ax.set_ylabel(ylabel)
        ax.set_title(titolo, fontsize=11)
        ax.set_ylim(0, max(valori or [1]) * 1.15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7); ax.tick_params(axis='x', rotation=60, labelsize=8)
        info_text = f"Ruote An: {ruote_analisi_str} | Spia: {numeri_spia_str} | Trigger: {risultato.get('totale_trigger', 0)}"
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=8)
        plt.tight_layout(pad=3.5); return fig
    except Exception as e:
        print(f"Errore in crea_grafico_barre ({tipo_analisi}, {tipo_stat}): {e}"); traceback.print_exc()
        if 'fig' in locals() and fig is not None: plt.close(fig)
        return None

def crea_tabella_lotto(risultato, info_ricerca, tipo_analisi="estratto", tipo_stat="presenza"):
    if tipo_analisi != "estratto": return None
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        ruota_verifica = info_ricerca.get('ruota_verifica', 'N/D')
        spia_val = info_ricerca.get('numeri_spia_input', ['N/D'])
        if isinstance(spia_val, list): numeri_spia_str = ", ".join(map(str,spia_val))
        elif isinstance(spia_val, tuple): numeri_spia_str = "-".join(map(str,spia_val))
        else: numeri_spia_str = str(spia_val)
        ruote_analisi_str = ", ".join(info_ricerca.get('ruote_analisi', []))
        n_trigger = risultato.get('totale_trigger', 0)
        numeri_lotto = np.arange(1, 91).reshape(9, 10)
        res_estratti = risultato.get('estratto', {})
        if tipo_stat == "presenza":
            percentuali_serie = res_estratti.get('all_percentuali_presenza', pd.Series(dtype='float64'))
            titolo = f"Tabella Lotto - Presenza ESTRATTI su {ruota_verifica}\n(dopo Spia {numeri_spia_str} su {ruote_analisi_str})"
        else:
            percentuali_serie = res_estratti.get('all_percentuali_frequenza', pd.Series(dtype='float64'))
            titolo = f"Tabella Lotto - Frequenza ESTRATTI su {ruota_verifica}\n(dopo Spia {numeri_spia_str} su {ruote_analisi_str})"
        percentuali = percentuali_serie.to_dict() if not percentuali_serie.empty else {}
        colors_norm = np.full(numeri_lotto.shape, 0.9)
        valid_perc = [p for p in percentuali.values() if pd.notna(p) and p > 0]
        max_perc = max(valid_perc) if valid_perc else 1.0
        if max_perc == 0: max_perc = 1.0
        for i in range(9):
            for j in range(10):
                num = numeri_lotto[i, j]; num_str = str(num).zfill(2); num_str_alt = str(num)
                perc_val = percentuali.get(num_str)
                if perc_val is None: perc_val = percentuali.get(num_str_alt)
                if perc_val is not None and pd.notna(perc_val) and perc_val > 0: colors_norm[i, j] = 0.9 - (0.9 * (perc_val / max_perc))
        for r_idx in range(10):
            ax.axvline(r_idx - 0.5, color='gray', linestyle='-', alpha=0.3); ax.axhline(r_idx - 0.5, color='gray', linestyle='-', alpha=0.3)
        for i in range(9):
            for j in range(10):
                num = numeri_lotto[i, j]; norm_color = colors_norm[i, j]
                cell_color = "white"; text_color = "black"
                if norm_color < 0.9:
                    intensity = (0.9 - norm_color) / 0.9
                    r_val = int(220 * (1 - intensity)); g_val = int(230 * (1 - intensity)); b_val = int(255 * (1 - intensity / 2))
                    cell_color = f"#{r_val:02x}{g_val:02x}{b_val:02x}"
                    if intensity > 0.6: text_color = "white"
                edge_color = 'black' if norm_color < 0.9 else 'gray'
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color=cell_color, alpha=1.0, edgecolor=edge_color, linewidth=1)
                ax.add_patch(rect); ax.text(j, i, num, ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")
        ax.set_xlim(-0.5, 9.5); ax.set_ylim(8.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        plt.title(titolo, fontsize=14, pad=15)
        info_text = f"Ruote An: {ruote_analisi_str} | Spia: {numeri_spia_str} | Trigger: {n_trigger}"
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]); return fig
    except Exception as e:
        print(f"Errore in crea_tabella_lotto: {e}"); traceback.print_exc()
        if 'fig' in locals() and fig is not None: plt.close(fig)
        return None

def crea_heatmap_correlazione(risultati, info_ricerca, tipo_analisi="estratto", tipo_stat="presenza"):
    if tipo_analisi != "estratto": return None
    fig = None
    try:
        if len(risultati) < 2: return None
        spia_val = info_ricerca.get('numeri_spia_input', ['N/D'])
        if isinstance(spia_val, list): numeri_spia_str = ", ".join(map(str,spia_val))
        elif isinstance(spia_val, tuple): numeri_spia_str = "-".join(map(str,spia_val))
        else: numeri_spia_str = str(spia_val)
        ruote_analisi_str = ", ".join(info_ricerca.get('ruote_analisi', []))
        all_numeri_estratti = set(); percentuali_per_ruota = {}
        for ruota_v, _, res in risultati:
            if res is None or not isinstance(res, dict): continue
            res_tipo = res.get('estratto', {});
            perc_dict_serie = res_tipo.get(f'all_percentuali_{tipo_stat}', pd.Series(dtype='float64'))
            if perc_dict_serie.empty: continue
            perc_dict = perc_dict_serie.to_dict()
            if not perc_dict: continue
            percentuali_per_ruota[ruota_v] = {num: perc for num, perc in perc_dict.items() if pd.notna(perc) and perc > 0}
            if percentuali_per_ruota[ruota_v]: all_numeri_estratti.update(percentuali_per_ruota[ruota_v].keys())
        if not percentuali_per_ruota or not all_numeri_estratti: return None
        all_numeri_sorted = sorted(list(all_numeri_estratti), key=str); ruote_heatmap = sorted(list(percentuali_per_ruota.keys()))
        if len(ruote_heatmap) < 2: return None
        matrice = np.array([[percentuali_per_ruota[r].get(n, np.nan) for n in all_numeri_sorted] for r in ruote_heatmap])
        fig, ax = plt.subplots(figsize=(min(18, len(all_numeri_sorted)*0.5+2), max(4, len(ruote_heatmap)*0.4+1)))
        max_val = np.nanmax(matrice) if not np.all(np.isnan(matrice)) else 1.0; max_val = max(max_val, 1.0)
        if SEABORN_AVAILABLE:
            sns.heatmap(matrice, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=all_numeri_sorted, yticklabels=ruote_heatmap, ax=ax, linewidths=.5, linecolor='gray', cbar=True, vmin=0, vmax=max_val, annot_kws={"size":7})
            ax.tick_params(axis='x',rotation=90,labelsize=8); ax.tick_params(axis='y',rotation=0,labelsize=8)
        else:
            cmap = plt.get_cmap("YlGnBu"); cmap.set_bad(color='lightgray'); im = ax.imshow(matrice,cmap=cmap,vmin=0,vmax=max_val,aspect='auto')
            ax.set_xticks(np.arange(len(all_numeri_sorted))); ax.set_yticks(np.arange(len(ruote_heatmap)))
            ax.set_xticklabels(all_numeri_sorted,rotation=90,fontsize=7); ax.set_yticklabels(ruote_heatmap,fontsize=7)
            for i_h in range(len(ruote_heatmap)):
                for j_h in range(len(all_numeri_sorted)):
                    val_h = matrice[i_h,j_h]
                    if not np.isnan(val_h): ax.text(j_h,i_h,f"{val_h:.1f}",ha="center",va="center",color="black" if val_h < 0.7*max_val else "white",fontsize=6)
            cbar = fig.colorbar(im,ax=ax,shrink=0.7); cbar.ax.set_ylabel(f"% {tipo_stat.capitalize()}",rotation=-90,va="bottom",fontsize=8); cbar.ax.tick_params(labelsize=7)
        plt.title(f"Heatmap {tipo_stat.capitalize()} ESTRATTI\n(Spia {numeri_spia_str} su {ruote_analisi_str})",fontsize=11,pad=15); plt.xlabel("Numeri Estratti",fontsize=9); plt.ylabel("Ruote Verifica",fontsize=9)
        plt.tight_layout(); return fig
    except Exception as e:
        print(f"Errore in crea_heatmap_correlazione: {e}"); traceback.print_exc()
        if 'fig' in locals() and fig is not None: plt.close(fig)
        return None

def visualizza_grafici(risultati_da_visualizzare, info_globale_ricerca, n_estrazioni_usate):
    try:
        if not risultati_da_visualizzare: messagebox.showinfo("Nessun Risultato", "Nessun risultato valido da visualizzare."); return
        win = tk.Toplevel(); win.title("Visualizzazione Grafica Risultati"); win.geometry("1300x850"); win.minsize(900,600)
        notebook = ttk.Notebook(win); notebook.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)
        def create_scrollable_tab(parent, tab_name):
            tab = ttk.Frame(parent); parent.add(tab, text=tab_name)
            canvas = tk.Canvas(tab); scrollbar_y = ttk.Scrollbar(tab,orient="vertical",command=canvas.yview)
            scrollbar_x = ttk.Scrollbar(tab,orient="horizontal",command=canvas.xview)
            scrollable_frame = ttk.Frame(canvas)
            scrollable_frame.bind("<Configure>",lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0,0),window=scrollable_frame,anchor="nw")
            canvas.configure(yscrollcommand=scrollbar_y.set,xscrollcommand=scrollbar_x.set)
            canvas.pack(side="top",fill="both",expand=True); scrollbar_y.pack(side="right",fill="y"); scrollbar_x.pack(side="bottom",fill="x"); return scrollable_frame
        barre_est_frame = create_scrollable_tab(notebook, "Grafici Barre (Estratti)")
        barre_amb_frame = create_scrollable_tab(notebook, "Grafici Barre (Ambi)")
        barre_ter_frame = create_scrollable_tab(notebook, "Grafici Barre (Terni)")
        tabelle_frame = create_scrollable_tab(notebook, "Tabelle Lotto (Solo Estratti)")
        heatmap_frame = create_scrollable_tab(notebook, "Heatmap Incrociata (Solo Estratti)")
        for frame, label_text in [(barre_est_frame, "Grafici a Barre per ESTRATTI Successivi"),
                                   (barre_amb_frame, "Grafici a Barre per AMBI Successivi"),
                                   (barre_ter_frame, "Grafici a Barre per TERNI Successivi"),
                                   (tabelle_frame, "Tabelle Lotto per ESTRATTI Successivi"),
                                   (heatmap_frame, "Heatmap Incrociata ESTRATTI tra Ruote di Verifica")]:
            ttk.Label(frame, text=label_text, style="Header.TLabel").pack(pady=10)
        for ruota_v, _, risultato in risultati_da_visualizzare:
            if risultato:
                info_specifica = info_globale_ricerca.copy(); info_specifica['ruota_verifica'] = ruota_v
                for tipo_an, frame_an in [("estratto", barre_est_frame), ("ambo", barre_amb_frame), ("terno", barre_ter_frame)]:
                    ruota_bar_frame_an = ttk.LabelFrame(frame_an, text=f"Ruota Verifica: {ruota_v} - {tipo_an.upper()}");
                    ruota_bar_frame_an.pack(fill="x",expand=False,padx=10,pady=10)
                    fig_p_an = crea_grafico_barre(risultato,info_specifica,tipo_an,"presenza")
                    fig_f_an = crea_grafico_barre(risultato,info_specifica,tipo_an,"frequenza")
                    if fig_p_an: FigureCanvasTkAgg(fig_p_an, master=ruota_bar_frame_an).get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
                    if fig_f_an: FigureCanvasTkAgg(fig_f_an, master=ruota_bar_frame_an).get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
                    if not fig_p_an and not fig_f_an: ttk.Label(ruota_bar_frame_an,text=f"Nessun grafico {tipo_an} generato").pack(padx=5,pady=5)
                ruota_tab_frame = ttk.LabelFrame(tabelle_frame, text=f"Ruota Verifica: {ruota_v}"); ruota_tab_frame.pack(fill="x",expand=False,padx=10,pady=10)
                fig_tp = crea_tabella_lotto(risultato,info_specifica,"estratto","presenza"); fig_tf = crea_tabella_lotto(risultato,info_specifica,"estratto","frequenza")
                if fig_tp: FigureCanvasTkAgg(fig_tp, master=ruota_tab_frame).get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
                if fig_tf: FigureCanvasTkAgg(fig_tf, master=ruota_tab_frame).get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
                if not fig_tp and not fig_tf: ttk.Label(ruota_tab_frame,text="Nessuna tabella generata").pack(padx=5,pady=5)
        risultati_validi_heatmap = [r for r in risultati_da_visualizzare if r[2] and isinstance(r[2],dict) and r[2].get('estratto')]
        if len(risultati_validi_heatmap) < 2: ttk.Label(heatmap_frame,text="Heatmap richiede >= 2 Ruote Verifica con risultati validi per gli estratti.").pack(padx=20,pady=20)
        else:
            for tipo_h_stat, nome_h_stat in [("presenza","Presenza"),("frequenza","Frequenza")]:
                heatmap_fig = crea_heatmap_correlazione(risultati_validi_heatmap,info_globale_ricerca,"estratto",tipo_h_stat)
                if heatmap_fig:
                    ttk.Label(heatmap_frame,text=f"--- Heatmap {nome_h_stat} (Estratti) ---",font=("Helvetica",11,"bold")).pack(pady=(10,5))
                    FigureCanvasTkAgg(heatmap_fig,master=heatmap_frame).get_tk_widget().pack(fill=tk.BOTH,expand=True,padx=5,pady=5)
                else: ttk.Label(heatmap_frame,text=f"Nessuna Heatmap {nome_h_stat} (Estratti) generata.").pack(pady=10)
    except Exception as e: messagebox.showerror("Errore Visualizzazione",f"Errore creazione finestra grafici:\n{e}"); traceback.print_exc()
    finally: plt.close('all')


# =============================================================================
# FUNZIONI LOGICHE (INVARIATE NELLA LORO LOGICA PRINCIPALE)
# =============================================================================
def carica_dati(file_path, start_date=None, end_date=None):
    try:
        if not os.path.exists(file_path): return None
        with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        dates, ruote, numeri, seen_rows, fmt_ok = [], [], [], set(), '%Y/%m/%d'
        for line in lines:
            line = line.strip();
            if not line: continue
            parts = line.split();
            if len(parts) < 7: continue
            data_str, ruota_str, nums_orig = parts[0], parts[1].upper(), parts[2:7]
            try:
                data_dt_val = datetime.datetime.strptime(data_str, fmt_ok)
                if len(nums_orig) != 5: continue
                [int(n) for n in nums_orig]
            except ValueError: continue

            if start_date and end_date and (data_dt_val.date() < start_date.date() or data_dt_val.date() > end_date.date()): continue
            key = f"{data_str}_{ruota_str}";
            if key in seen_rows: continue
            seen_rows.add(key); dates.append(data_str); ruote.append(ruota_str); numeri.append(nums_orig)
        if not dates: return None
        df = pd.DataFrame({'Data': dates, 'Ruota': ruote, **{f'Numero{i+1}': [n[i] for n in numeri] for i in range(5)}})
        df['Data'] = pd.to_datetime(df['Data'], format=fmt_ok)
        for col in [f'Numero{i+1}' for i in range(5)]:
             df[col] = df[col].apply(lambda x: str(int(x)).zfill(2) if pd.notna(x) and str(x).isdigit() and 1 <= int(x) <= 90 else pd.NA)
        df.dropna(subset=[f'Numero{i+1}' for i in range(5)], how='any', inplace=True)
        df = df.sort_values(by='Data').reset_index(drop=True)
        return df if not df.empty else None
    except Exception as e:
        print(f"Errore lettura file {os.path.basename(file_path)}: {e}"); traceback.print_exc(); return None

def analizza_ruota_verifica(df_verifica, date_trigger_sorted, n_estrazioni, nome_ruota_verifica):
    if df_verifica is None or df_verifica.empty: return None, "Df verifica vuoto."
    df_verifica = df_verifica.sort_values(by='Data').drop_duplicates(subset=['Data']).reset_index(drop=True)
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']
    n_trigger = len(date_trigger_sorted)
    date_series_verifica = df_verifica['Data']

    freq_estratti, freq_ambi, freq_terne = Counter(), Counter(), Counter()
    pres_estratti, pres_ambi, pres_terne = Counter(), Counter(), Counter()

    ambo_copertura_trigger = Counter()
    freq_pos_estratti = {}
    pres_pos_estratti = {}

    for data_t in date_trigger_sorted:
        try:
            start_index = date_series_verifica.searchsorted(data_t, side='right')
        except Exception:
            continue
        if start_index >= len(date_series_verifica):
            continue

        df_successive = df_verifica.iloc[start_index : start_index + n_estrazioni]
        estratti_unici_finestra, ambi_unici_finestra, terne_unici_finestra = set(), set(), set()
        estratti_pos_unici_finestra = {}

        if not df_successive.empty:
            for _, row in df_successive.iterrows():
                numeri_estratti_riga_con_pos = []
                for pos_idx, col_num_nome in enumerate(colonne_numeri):
                    num_val = row[col_num_nome]
                    if pd.notna(num_val):
                        numeri_estratti_riga_con_pos.append((str(num_val).zfill(2), pos_idx + 1))

                if not numeri_estratti_riga_con_pos: continue

                for num_str, pos in numeri_estratti_riga_con_pos:
                    freq_estratti[num_str] += 1
                    estratti_unici_finestra.add(num_str)

                    if num_str not in freq_pos_estratti:
                        freq_pos_estratti[num_str] = Counter()
                    freq_pos_estratti[num_str][pos] += 1

                    if num_str not in estratti_pos_unici_finestra:
                        estratti_pos_unici_finestra[num_str] = set()
                    estratti_pos_unici_finestra[num_str].add(pos)

                numeri_solo_per_combinazioni = sorted([item[0] for item in numeri_estratti_riga_con_pos])
                if len(numeri_solo_per_combinazioni) >= 2:
                    for ambo in itertools.combinations(numeri_solo_per_combinazioni, 2):
                        ambo_ordinato = tuple(sorted(ambo))
                        freq_ambi[ambo_ordinato] += 1
                        ambi_unici_finestra.add(ambo_ordinato)

                if len(numeri_solo_per_combinazioni) >= 3:
                    for terno in itertools.combinations(numeri_solo_per_combinazioni, 3):
                        terno_ordinato = tuple(sorted(terno))
                        freq_terne[terno_ordinato] += 1
                        terne_unici_finestra.add(terno_ordinato)

        for num in estratti_unici_finestra: pres_estratti[num] += 1
        for ambo_u in ambi_unici_finestra:
            pres_ambi[ambo_u] += 1
            ambo_copertura_trigger[ambo_u] +=1
        for terno in terne_unici_finestra: pres_terne[terno] += 1

        for num_str, pos_set in estratti_pos_unici_finestra.items():
            if num_str not in pres_pos_estratti:
                pres_pos_estratti[num_str] = Counter()
            for pos_val in pos_set:
                pres_pos_estratti[num_str][pos_val] +=1

    results = {'totale_trigger': n_trigger}
    for tipo, freq_dict, pres_dict in [('estratto', freq_estratti, pres_estratti),
                                       ('ambo', freq_ambi, pres_ambi),
                                       ('terno', freq_terne, pres_terne)]:
        if not freq_dict:
            results[tipo] = {'presenza': {'top':pd.Series(dtype=int),'percentuali':pd.Series(dtype=float),'frequenze':pd.Series(dtype=int),'perc_frequenza':pd.Series(dtype=float)},
                             'frequenza':{'top':pd.Series(dtype=int),'percentuali':pd.Series(dtype=float),'presenze':pd.Series(dtype=int),'perc_presenza':pd.Series(dtype=float)},
                             'all_percentuali_presenza':pd.Series(dtype=float), 'all_percentuali_frequenza':pd.Series(dtype=float),
                             'full_presenze':pd.Series(dtype=int), 'full_frequenze':pd.Series(dtype=int)}
            if tipo == 'estratto':
                 results[tipo]['posizionale_frequenza'] = {}
                 results[tipo]['posizionale_presenza'] = {}
            if tipo == 'ambo':
                results[tipo]['migliori_per_copertura_trigger'] = {
                    'items': [],
                    'totale_trigger_spia': n_trigger
                }
            continue

        if tipo in ['ambo', 'terno']:
            freq_s = pd.Series({k: v for k,v in freq_dict.items()}, dtype=int).sort_index()
            pres_s = pd.Series({k: v for k,v in pres_dict.items()}, dtype=int)
        else:
            freq_s = pd.Series(freq_dict, dtype=int).sort_index()
            pres_s = pd.Series(pres_dict, dtype=int)

        pres_s = pres_s.reindex(freq_s.index, fill_value=0).sort_index()

        tot_freq = freq_s.sum()
        perc_freq = (freq_s / tot_freq * 100).round(2) if tot_freq > 0 else pd.Series(0.0, index=freq_s.index, dtype=float)
        perc_pres = (pres_s / n_trigger * 100).round(2) if n_trigger > 0 else pd.Series(0.0, index=pres_s.index, dtype=float)

        top_pres_items = pres_s.sort_values(ascending=False).head(10)
        top_freq_items = freq_s.sort_values(ascending=False).head(10)

        if tipo in ['ambo', 'terno']:
            top_pres_formatted = pd.Series({format_ambo_terno(k): v for k,v in top_pres_items.items()}, dtype=int)
            top_freq_formatted = pd.Series({format_ambo_terno(k): v for k,v in top_freq_items.items()}, dtype=int)
        else:
            top_pres_formatted = top_pres_items
            top_freq_formatted = top_freq_items

        results[tipo] = {
            'presenza': {'top': top_pres_formatted,
                         'percentuali': perc_pres.reindex(top_pres_items.index).rename(index=format_ambo_terno if tipo in ['ambo','terno'] else None).fillna(0.0),
                         'frequenze': freq_s.reindex(top_pres_items.index).rename(index=format_ambo_terno if tipo in ['ambo','terno'] else None).fillna(0).astype(int),
                         'perc_frequenza': perc_freq.reindex(top_pres_items.index).rename(index=format_ambo_terno if tipo in ['ambo','terno'] else None).fillna(0.0)},
            'frequenza': {'top': top_freq_formatted,
                          'percentuali': perc_freq.reindex(top_freq_items.index).rename(index=format_ambo_terno if tipo in ['ambo','terno'] else None).fillna(0.0),
                          'presenze': pres_s.reindex(top_freq_items.index).rename(index=format_ambo_terno if tipo in ['ambo','terno'] else None).fillna(0).astype(int),
                          'perc_presenza': perc_pres.reindex(top_pres_items.index).rename(index=format_ambo_terno if tipo in ['ambo','terno'] else None).fillna(0.0)},
            'all_percentuali_presenza': perc_pres.rename(index=format_ambo_terno if tipo in ['ambo','terno'] else None),
            'all_percentuali_frequenza': perc_freq.rename(index=format_ambo_terno if tipo in ['ambo','terno'] else None),
            'full_presenze': pres_s.rename(index=format_ambo_terno if tipo in ['ambo','terno'] else None),
            'full_frequenze': freq_s.rename(index=format_ambo_terno if tipo in ['ambo','terno'] else None)
        }

    if 'ambo' in results and isinstance(results['ambo'], dict):
        if n_trigger > 0 and ambo_copertura_trigger:
            migliori_ambi_copertura_raw = sorted(
                ambo_copertura_trigger.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True
            )
            top_ambi_copertura_list = []
            if migliori_ambi_copertura_raw:
                top_ambi_copertura_list = [
                   (format_ambo_terno(ambo_tuple), count)
                   for ambo_tuple, count in migliori_ambi_copertura_raw
                ][:3]

            results['ambo']['migliori_per_copertura_trigger'] = {
                'items': top_ambi_copertura_list,
                'totale_trigger_spia': n_trigger
            }
        else:
            results['ambo']['migliori_per_copertura_trigger'] = {
                'items': [],
                'totale_trigger_spia': n_trigger
            }

    if 'estratto' in results and freq_pos_estratti:
        results['estratto']['posizionale_frequenza'] = {
            num: dict(sorted(pos_counts.items())) for num, pos_counts in freq_pos_estratti.items()
        }
        results['estratto']['posizionale_presenza'] = {
            num: dict(sorted(pos_counts.items())) for num, pos_counts in pres_pos_estratti.items()
        }
    elif 'estratto' in results:
        results['estratto']['posizionale_frequenza'] = {}
        results['estratto']['posizionale_presenza'] = {}

    return (results, None) if any(results.get(t) for t in ['estratto', 'ambo', 'terno']) else (None, f"Nessun risultato su {nome_ruota_verifica}.")

def analizza_antecedenti(df_ruota, numeri_obiettivo, n_precedenti, nome_ruota):
    if df_ruota is None or df_ruota.empty: return None, "DataFrame vuoto."
    if not numeri_obiettivo or n_precedenti <= 0: return None, "Input invalidi."
    df_ruota = df_ruota.sort_values(by='Data').reset_index(drop=True); cols_num = [f'Numero{i+1}' for i in range(5)]
    numeri_obiettivo_zfill = [str(n).zfill(2) for n in numeri_obiettivo]

    indices_obiettivo = df_ruota.index[df_ruota[cols_num].isin(numeri_obiettivo_zfill).any(axis=1)].tolist()
    n_occ_obiettivo = len(indices_obiettivo)
    if n_occ_obiettivo == 0: return None, f"Obiettivi non trovati su {nome_ruota}."
    freq_ant, pres_ant, actual_base_pres = Counter(), Counter(), 0
    for idx_obj in indices_obiettivo:
        if idx_obj < n_precedenti: continue
        actual_base_pres += 1
        df_prec = df_ruota.iloc[idx_obj - n_precedenti : idx_obj]
        if not df_prec.empty:
            numeri_finestra_unici = set()
            for _, row_prec in df_prec.iterrows():
                estratti_prec_riga = [row_prec[col] for col in cols_num if pd.notna(row_prec[col])]
                freq_ant.update(estratti_prec_riga)
                numeri_finestra_unici.update(estratti_prec_riga)
            pres_ant.update(list(numeri_finestra_unici))

    empty_stats = lambda: {'top':pd.Series(dtype=int),'percentuali':pd.Series(dtype=float),'frequenze':pd.Series(dtype=int),'perc_frequenza':pd.Series(dtype=float)}
    empty_freq_stats = lambda: {'top':pd.Series(dtype=int),'percentuali':pd.Series(dtype=float),'presenze':pd.Series(dtype=int),'perc_presenza':pd.Series(dtype=float)}
    base_res = {'totale_occorrenze_obiettivo':n_occ_obiettivo, 'base_presenza_antecedenti':actual_base_pres, 'numeri_obiettivo':numeri_obiettivo_zfill, 'n_precedenti':n_precedenti, 'nome_ruota':nome_ruota}

    if actual_base_pres == 0 or not freq_ant:
        return {**base_res, 'presenza':empty_stats(), 'frequenza':empty_freq_stats()}, "Nessuna finestra/numero antecedente valido."

    ant_freq_s = pd.Series(freq_ant, dtype=int).sort_index()
    ant_pres_s = pd.Series(pres_ant, dtype=int).reindex(ant_freq_s.index, fill_value=0).sort_index()

    tot_ant_freq = ant_freq_s.sum(); perc_ant_freq = (ant_freq_s/tot_ant_freq*100).round(2) if tot_ant_freq > 0 else pd.Series(0.0, index=ant_freq_s.index)
    perc_ant_pres = (ant_pres_s/actual_base_pres*100).round(2) if actual_base_pres > 0 else pd.Series(0.0, index=ant_pres_s.index)

    top_ant_pres = ant_pres_s.sort_values(ascending=False).head(10)
    top_ant_freq = ant_freq_s.sort_values(ascending=False).head(10)

    return {**base_res,
            'presenza': {'top':top_ant_pres, 'percentuali':perc_ant_pres.reindex(top_ant_pres.index).fillna(0.0),
                         'frequenze':ant_freq_s.reindex(top_ant_pres.index).fillna(0).astype(int),
                         'perc_frequenza':perc_ant_freq.reindex(top_ant_pres.index).fillna(0.0)},
            'frequenza':{'top':top_ant_freq, 'percentuali':perc_ant_freq.reindex(top_ant_freq.index).fillna(0.0),
                         'presenze':ant_pres_s.reindex(top_ant_freq.index).fillna(0).astype(int),
                         'perc_presenza':perc_ant_pres.reindex(top_ant_freq.index).fillna(0.0)}
           }, None

def aggiorna_risultati_globali(risultati_nuovi, info_ricerca=None, modalita="successivi"):
    global risultati_globali, info_ricerca_globale, button_visualizza, button_verifica_futura, button_verifica_mista

    if button_visualizza: button_visualizza.config(state=tk.DISABLED)
    if button_verifica_futura: button_verifica_futura.config(state=tk.DISABLED)
    if button_verifica_mista: button_verifica_mista.config(state=tk.DISABLED)

    if modalita == "successivi":
        risultati_globali = risultati_nuovi if risultati_nuovi is not None else []
        info_ricerca_globale = info_ricerca if info_ricerca is not None else {}
        has_valid_results = bool(risultati_globali) and any(res[2] for res in risultati_globali if len(res)>2)
        has_date_trigger = bool(info_ricerca_globale.get('date_trigger_ordinate'))
        has_end_date = info_ricerca_globale.get('end_date') is not None
        has_ruote_verifica_info = bool(info_ricerca_globale.get('ruote_verifica'))

        if has_valid_results and button_visualizza: button_visualizza.config(state=tk.NORMAL)
        if has_end_date and has_ruote_verifica_info and button_verifica_futura: button_verifica_futura.config(state=tk.NORMAL)
        if has_date_trigger and has_ruote_verifica_info and button_verifica_mista: button_verifica_mista.config(state=tk.NORMAL)
    else:
        risultati_globali, info_ricerca_globale = [], {}

# MODIFICATO per non alterare permanentemente lo stato di risultato_text
def salva_risultati():
    global risultato_text, root
    content = risultato_text.get(1.0, tk.END).strip()
    if not content or any(msg in content for msg in ["Benvenuto", "Ricerca in corso...", "Nessun risultato", "Elaborazione in corso..."]):
        messagebox.showinfo("Salvataggio", "Nessun risultato significativo da salvare.", parent=root)
        return
    fpath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], title="Salva Risultati")
    if fpath:
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Salvataggio OK", f"Risultati salvati in:\n{fpath}", parent=root)
        except Exception as e:
            messagebox.showerror("Errore Salvataggio", f"Impossibile salvare il file:\n{e}", parent=root)

def format_ambo_terno(combinazione):
    if isinstance(combinazione, tuple) or isinstance(combinazione, list):
        return "-".join(map(str, combinazione))
    return str(combinazione)

def calcola_ritardo_attuale(df_ruota_completa, item_da_cercare, tipo_item, data_fine_analisi):
    """
    Calcola il ritardo attuale di un estratto o ambo su una specifica ruota
    fino a una data di fine analisi.
    """
    if df_ruota_completa is None or df_ruota_completa.empty:
        return "N/D (no data)"
    if not isinstance(data_fine_analisi, pd.Timestamp):
        data_fine_analisi = pd.Timestamp(data_fine_analisi)

    # Filtra le estrazioni fino alla data di fine analisi e ordina per data decrescente
    df_filtrato = df_ruota_completa[df_ruota_completa['Data'] <= data_fine_analisi].sort_values(by='Data', ascending=False)

    if df_filtrato.empty:
        return "N/D (no draws in period)"

    ritardo = 0
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']

    for _, row in df_filtrato.iterrows():
        ritardo += 1
        numeri_riga = {str(row[col]).zfill(2) for col in colonne_numeri if pd.notna(row[col])}

        trovato = False
        if tipo_item == "estratto":
            if isinstance(item_da_cercare, str) and item_da_cercare in numeri_riga:
                trovato = True
        elif tipo_item == "ambo":
            if isinstance(item_da_cercare, tuple) and len(item_da_cercare) == 2:
                if set(item_da_cercare).issubset(numeri_riga):
                    trovato = True

        if trovato:
            return ritardo -1 # -1 perché il ritardo è 0 se esce nell'ultima estrazione considerata

    return ritardo # Se non trovato, il ritardo è il numero totale di estrazioni analizzate


# MODIFICATO per includere pulsante "Salva su File..." e correggere struttura
def mostra_popup_testo_semplice(titolo, contenuto_testo):
    global root
    popup_window = tk.Toplevel(root)
    popup_window.title(titolo)

    num_righe = contenuto_testo.count('\n') + 1
    max_len_riga = 0
    if contenuto_testo:
        try: max_len_riga = max(len(r) for r in contenuto_testo.split('\n'))
        except ValueError: max_len_riga = 80
    else: max_len_riga = 80

    larghezza_stimata_char = max(80, min(120, max_len_riga + 5))
    altezza_stimata_righe = max(15, min(40, num_righe + 5))
    popup_width = larghezza_stimata_char * 7
    popup_height = altezza_stimata_righe * 16
    popup_width = max(500, min(900, popup_width))
    popup_height = max(300, min(700, popup_height))

    popup_window.geometry(f"{int(popup_width)}x{int(popup_height)}")
    popup_window.transient(root)

    text_widget = scrolledtext.ScrolledText(popup_window, wrap=tk.WORD, font=("Consolas", 10))
    text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    text_widget.insert(tk.END, contenuto_testo)
    text_widget.config(state=tk.DISABLED) # Permette la selezione e la copia standard

    button_frame_popup = ttk.Frame(popup_window)
    button_frame_popup.pack(fill=tk.X, pady=(0,10), padx=10, side=tk.BOTTOM)

    # Funzione interna per salvare il contenuto del popup
    def _salva_popup_content_locale():
        fpath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Salva contenuto popup"
        )
        if fpath:
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(contenuto_testo)
                messagebox.showinfo("Salvataggio OK", f"Contenuto del popup salvato in:\n{fpath}", parent=popup_window)
            except Exception as e_save:
                messagebox.showerror("Errore Salvataggio", f"Impossibile salvare il file:\n{e_save}", parent=popup_window)

    ttk.Button(button_frame_popup, text="Salva su File...", command=_salva_popup_content_locale).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame_popup, text="Chiudi", command=popup_window.destroy).pack(side=tk.RIGHT, padx=5)

    popup_window.update_idletasks()
    master_x = root.winfo_x(); master_y = root.winfo_y()
    master_width = root.winfo_width(); master_height = root.winfo_height()
    win_width = popup_window.winfo_width(); win_height = popup_window.winfo_height()
    center_x = master_x + (master_width // 2) - (win_width // 2)
    center_y = master_y + (master_height // 2) - (win_height // 2)
    popup_window.geometry(f"+{center_x}+{center_y}")


# MODIFICATO per includere pulsante "Salva su File..." (UNA VOLTA) e RITARDI (per ruota e combinati e globali)
def mostra_popup_risultati_spia(info_ricerca, risultati_analisi):
    global root
    popup = tk.Toplevel(root)
    popup.title("Riepilogo Analisi Numeri Spia")
    popup.geometry("850x800")
    popup.transient(root)

    text_area_popup = scrolledtext.ScrolledText(popup, wrap=tk.WORD, font=("Consolas", 10), state=tk.DISABLED)
    text_area_popup.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10,0))

    popup_content_list = ["=== RIEPILOGO ANALISI NUMERI SPIA ===\n"]
    tipo_spia_usato = info_ricerca.get('tipo_spia_usato', 'N/D').upper()
    spia_val = info_ricerca.get('numeri_spia_input', [])
    spia_display = ""
    if tipo_spia_usato == "ESTRATTO_POSIZIONALE":
        num_spia_list = spia_val if isinstance(spia_val, list) else [spia_val]
        num_spia = num_spia_list[0] if num_spia_list else "N/D"
        pos_spia = info_ricerca.get('posizione_spia_input', "N/D")
        spia_display = f"{num_spia} in {pos_spia}a pos."
    elif isinstance(spia_val, tuple): spia_display = "-".join(map(str, spia_val))
    elif isinstance(spia_val, list): spia_display = ", ".join(map(str, spia_val))
    else: spia_display = str(spia_val)

    popup_content_list.append(f"Tipo Spia: {tipo_spia_usato.replace('_', ' ')} ({spia_display})")
    popup_content_list.append(f"Ruote Analisi (Spia): {', '.join(info_ricerca.get('ruote_analisi', []))}")
    popup_content_list.append(f"Ruote Verifica (Esiti): {', '.join(info_ricerca.get('ruote_verifica', []))}")
    start_date_pd = info_ricerca.get('start_date', pd.NaT); end_date_pd = info_ricerca.get('end_date', pd.NaT)
    start_date_str = start_date_pd.strftime('%d/%m/%Y') if pd.notna(start_date_pd) else "N/D"
    end_date_str = end_date_pd.strftime('%d/%m/%Y') if pd.notna(end_date_pd) else "N/D"
    popup_content_list.append(f"Periodo: {start_date_str} - {end_date_str}")
    popup_content_list.append(f"Estrazioni Successive Analizzate: {info_ricerca.get('n_estrazioni', 'N/D')}")
    date_triggers = info_ricerca.get('date_trigger_ordinate', [])
    popup_content_list.append(f"Numero Totale di Eventi Spia (Trigger): {len(date_triggers)}")
    popup_content_list.append("-" * 60)

    if not risultati_analisi:
        popup_content_list.append("\nNessun risultato dettagliato per le ruote di verifica.")
    else:
        for nome_ruota_v, _, res_ruota in risultati_analisi:
            if not res_ruota or not isinstance(res_ruota, dict): continue
            popup_content_list.append(f"\n\n--- RISULTATI PER RUOTA DI VERIFICA: {nome_ruota_v.upper()} ---")
            popup_content_list.append(f"(Basato su {res_ruota.get('totale_trigger', 0)} eventi spia)")
            for tipo_esito in ['estratto', 'ambo', 'terno']:
                dati_esito = res_ruota.get(tipo_esito)
                if dati_esito:
                    popup_content_list.append(f"\n  -- {tipo_esito.capitalize()} Successivi --")
                    popup_content_list.append("    Top per Presenza (su casi trigger):")
                    top_pres = dati_esito.get('presenza', {}).get('top')
                    if top_pres is not None and not top_pres.empty:
                        for item_str_key, pres_val in top_pres.items():
                            perc = dati_esito['presenza']['percentuali'].get(item_str_key, 0.0)
                            freq = dati_esito['presenza']['frequenze'].get(item_str_key, 0)
                            riga_base = f"      - {item_str_key}: {pres_val} ({perc:.1f}%) [Freq.Tot: {freq}]"
                            ritardo_str_popup = ""
                            if tipo_esito in ['estratto', 'ambo'] and 'ritardi_attuali' in dati_esito:
                                ritardo_val_popup = dati_esito['ritardi_attuali'].get(item_str_key)
                                if ritardo_val_popup is not None:
                                    ritardo_str_popup = f" [Rit.Att: {ritardo_val_popup}]"
                                else:
                                    ritardo_str_popup = f" [Rit.Att: N/D]"
                            popup_content_list.append(riga_base + ritardo_str_popup)
                            if tipo_esito == 'estratto' and 'posizionale_presenza' in dati_esito and item_str_key in dati_esito['posizionale_presenza']:
                                pos_data = dati_esito['posizionale_presenza'][item_str_key]; pos_str_list = []
                                for pos_num in sorted(pos_data.keys()):
                                    pos_count = pos_data[pos_num]; pos_perc = (pos_count / pres_val * 100) if pres_val > 0 else 0
                                    pos_str_list.append(f"P{pos_num}:{pos_count}({pos_perc:.0f}%)")
                                if pos_str_list: popup_content_list.append(f"        Posizioni (Pres.): {', '.join(pos_str_list)}")
                    else:
                        popup_content_list.append("      Nessuno.")

                    popup_content_list.append("    Top per Frequenza Totale:")
                    top_freq = dati_esito.get('frequenza', {}).get('top')
                    if top_freq is not None and not top_freq.empty:
                        for item_str_key, freq_val in top_freq.items():
                            perc = dati_esito['frequenza']['percentuali'].get(item_str_key, 0.0)
                            pres = dati_esito['frequenza']['presenze'].get(item_str_key, 0)
                            riga_base = f"      - {item_str_key}: {freq_val} ({perc:.1f}%) [Pres. su Trigger: {pres}]"
                            popup_content_list.append(riga_base)
                            if tipo_esito == 'estratto' and 'posizionale_frequenza' in dati_esito and item_str_key in dati_esito['posizionale_frequenza']:
                                pos_data = dati_esito['posizionale_frequenza'][item_str_key]; pos_str_list = []
                                for pos_num in sorted(pos_data.keys()):
                                    pos_count = pos_data[pos_num]; pos_perc = (pos_count / freq_val * 100) if freq_val > 0 else 0
                                    pos_str_list.append(f"P{pos_num}:{pos_count}({pos_perc:.0f}%)")
                                if pos_str_list: popup_content_list.append(f"        Posizioni (Freq.): {', '.join(pos_str_list)}")
                    else:
                        popup_content_list.append("      Nessuno.")
                    if tipo_esito == 'ambo':
                        migliori_ambi_cop_info = dati_esito.get('migliori_per_copertura_trigger')
                        if migliori_ambi_cop_info and migliori_ambi_cop_info['items']:
                            popup_content_list.append(f"    Migliori Ambi per Copertura Eventi Spia (su {migliori_ambi_cop_info['totale_trigger_spia']} totali):")
                            for ambo_str_popup, count_cop_popup in migliori_ambi_cop_info['items']:
                                perc_cop_popup = (count_cop_popup / migliori_ambi_cop_info['totale_trigger_spia'] * 100) if migliori_ambi_cop_info['totale_trigger_spia'] > 0 else 0
                                popup_content_list.append(f"      - Ambo {ambo_str_popup}: Coperti {count_cop_popup} eventi spia ({perc_cop_popup:.1f}%)")
                        elif dati_esito: popup_content_list.append(f"    Migliori Ambi per Copertura Eventi Spia: Nessuno con copertura significativa.")
                else:
                    popup_content_list.append(f"\n  -- {tipo_esito.capitalize()} Successivi: Nessun dato trovato.")

    statistiche_combinate_dett = info_ricerca.get('statistiche_combinate_dettagliate')
    if statistiche_combinate_dett and any(v for v in statistiche_combinate_dett.values() if v):
        popup_content_list.append("\n\n" + "=" * 25 + " RISULTATI COMBINATI (PER PUNTEGGIO) " + "=" * 25)
        for tipo_esito_comb in ['estratto', 'ambo', 'terno']:
            dati_tipo_comb_dett = statistiche_combinate_dett.get(tipo_esito_comb)
            if dati_tipo_comb_dett:
                popup_content_list.append(f"\n  -- Top {tipo_esito_comb.capitalize()} Combinati (per Punteggio) --")
                for i, stat_item in enumerate(dati_tipo_comb_dett):
                    item_str_c = stat_item["item"]
                    score_c = stat_item["punteggio"]
                    pres_avg_c = stat_item["presenza_media_perc"]
                    freq_tot_c = stat_item["frequenza_totale"]
                    ritardo_comb_str = ""
                    if tipo_esito_comb in ['estratto', 'ambo'] and "ritardo_min_attuale" in stat_item:
                        rit_val = stat_item["ritardo_min_attuale"]
                        if rit_val not in ["N/A", "N/D", None]:
                            ritardo_comb_str = f" [Rit.Min.Att: {rit_val}]"
                        elif rit_val == "N/D":
                             ritardo_comb_str = f" [Rit.Min.Att: N/D]"
                    popup_content_list.append(f"    {i+1}. {item_str_c}: Punt={score_c:.2f} (PresAvg:{pres_avg_c:.1f}%, FreqTot:{freq_tot_c}){ritardo_comb_str}")
            else:
                popup_content_list.append(f"\n  -- Top {tipo_esito_comb.capitalize()} Combinati: Nessuno.")
    elif info_ricerca.get('top_combinati') and any(v for v in info_ricerca.get('top_combinati').values() if v) :
        popup_content_list.append("\n\n" + "=" * 25 + " RISULTATI COMBINATI (SOLO ITEM) " + "=" * 25)
        top_combinati_fallback = info_ricerca.get('top_combinati')
        for tipo_esito_comb in ['estratto', 'ambo', 'terno']:
            if top_combinati_fallback.get(tipo_esito_comb):
                popup_content_list.append(f"\n  -- Top {tipo_esito_comb.capitalize()} Combinati (solo item) --")
                for item_comb in top_combinati_fallback[tipo_esito_comb][:10]:
                     popup_content_list.append(f"    - {format_ambo_terno(item_comb)}")
            else: popup_content_list.append(f"\n  -- Top {tipo_esito_comb.capitalize()} Combinati: Nessuno.")

    migliori_ambi_globali_info = info_ricerca.get('migliori_ambi_copertura_globale')
    if migliori_ambi_globali_info:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORI AMBI PER COPERTURA GLOBALE DEGLI EVENTI SPIA " + "=" * 10)
        popup_content_list.append(f"(Uscita su QUALSIASI ruota di verifica dopo ogni evento spia)")
        for i, ambo_info in enumerate(migliori_ambi_globali_info):
            rit_glob_str = ""
            if "ritardo_min_attuale" in ambo_info:
                rit_val_glob = ambo_info["ritardo_min_attuale"]
                if rit_val_glob not in ["N/A", "N/D", None]:
                    rit_glob_str = f" [Rit.Min.Att: {rit_val_glob}]"
                elif rit_val_glob == "N/D":
                     rit_glob_str = f" [Rit.Min.Att: N/D]"
            popup_content_list.append(f"  {i+1}. Ambo {ambo_info['ambo']}: Coperti {ambo_info['coperti']} su {ambo_info['totali']} eventi ({ambo_info['percentuale']:.1f}%){rit_glob_str}")

    combinazione_ottimale_info = info_ricerca.get('combinazione_ottimale_copertura_100')
    migliore_parziale_info = info_ricerca.get('migliore_combinazione_parziale')

    if combinazione_ottimale_info:
        popup_content_list.append("\n\n" + "=" * 10 + " COMBINAZIONE AMBI PER COPERTURA TOTALE " + "=" * 10)
        popup_content_list.append(f"  I seguenti {len(combinazione_ottimale_info.get('ambi_dettagli', combinazione_ottimale_info.get('ambi',[])))} ambo/i:")
        ambi_da_mostrare_ottimale = combinazione_ottimale_info.get("ambi_dettagli", combinazione_ottimale_info.get("ambi",[]))
        for ambo_s_combinazione_dett in ambi_da_mostrare_ottimale:
            popup_content_list.append(f"    - {ambo_s_combinazione_dett}")
        popup_content_list.append(f"  Insieme hanno coperto il 100% ({combinazione_ottimale_info['coperti']}/{combinazione_ottimale_info['totali']} eventi).")

    elif migliore_parziale_info:
        popup_content_list.append("\n\n" + "=" * 10 + " MIGLIORE COPERTURA COMBINATA (NON 100%) " + "=" * 10)
        popup_content_list.append(f"  Nessuna combinazione di 1, 2 o 3 ambi (dai top) ha raggiunto il 100%.")
        ambi_da_visualizzare_popup_parziale = migliore_parziale_info.get("ambi_dettagli", migliore_parziale_info.get("ambi",[]))
        popup_content_list.append(f"  Considerando i Top {len(migliore_parziale_info.get('ambi',[]))} ambi:")
        for ambo_s_parziale_dett in ambi_da_visualizzare_popup_parziale:
            popup_content_list.append(f"    - {ambo_s_parziale_dett}")
        popup_content_list.append(f"  Copertura combinata: {migliore_parziale_info['coperti']}/{migliore_parziale_info['totali']} eventi ({migliore_parziale_info['percentuale']:.1f}%).")

    elif info_ricerca.get('migliori_ambi_copertura_globale') is not None and not combinazione_ottimale_info and not migliore_parziale_info and 'date_trigger_ordinate' in info_ricerca and info_ricerca['date_trigger_ordinate']:
        popup_content_list.append("\n\n" + "=" * 10 + " COMBINAZIONE AMBI PER COPERTURA TOTALE " + "=" * 10)
        popup_content_list.append("  Non è stato possibile trovare una combinazione di 1-3 ambi (dai top) per la copertura totale,")
        popup_content_list.append("  o non sono stati trovati ambi con copertura globale significativa.")

    final_popup_text_content = "\n".join(popup_content_list)

    text_area_popup.config(state=tk.NORMAL)
    text_area_popup.delete(1.0, tk.END)
    text_area_popup.insert(tk.END, final_popup_text_content)
    text_area_popup.config(state=tk.DISABLED)

    button_frame_popup_spia = ttk.Frame(popup)
    button_frame_popup_spia.pack(fill=tk.X, pady=(5,10), padx=10, side=tk.BOTTOM)

    def _salva_popup_spia_content():
        fpath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Salva Riepilogo Analisi Spia"
        )
        if fpath:
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(final_popup_text_content)
                messagebox.showinfo("Salvataggio OK", f"Riepilogo salvato in:\n{fpath}", parent=popup)
            except Exception as e_save:
                messagebox.showerror("Errore Salvataggio", f"Impossibile salvare il file:\n{e_save}", parent=popup)

    ttk.Button(button_frame_popup_spia, text="Salva su File...", command=_salva_popup_spia_content).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame_popup_spia, text="Chiudi", command=popup.destroy).pack(side=tk.RIGHT, padx=5)

    popup.update_idletasks()
    master_x = root.winfo_x(); master_y = root.winfo_y()
    master_width = root.winfo_width(); master_height = root.winfo_height()
    win_width = popup.winfo_width(); win_height = popup.winfo_height()
    center_x = master_x + (master_width // 2) - (win_width // 2)
    center_y = master_y + (master_height // 2) - (win_height // 2)
    popup.geometry(f"+{center_x}+{center_y}")

# MODIFICATO per gestione stato risultato_text, RITARDI PER RUOTA, RITARDI COMBINATI MINIMI, RITARDI AMBI GLOBALI (CON CORREZIONE F-STRING)
def cerca_numeri(modalita="successivi"):
    global risultati_globali, info_ricerca_globale, file_ruote, risultato_text, root
    global start_date_entry, end_date_entry, listbox_ruote_analisi, listbox_ruote_verifica
    global entry_numeri_spia, estrazioni_entry_succ, listbox_ruote_analisi_ant
    global entry_numeri_obiettivo, estrazioni_entry_ant, tipo_spia_var_global, combo_posizione_spia
    global MAX_COLPI_GIOCO

    if not mappa_file_ruote() or not file_ruote:
        messagebox.showerror("Errore Cartella", "Impossibile leggere i file dalla cartella specificata o la cartella non contiene file validi.\nAssicurati di aver selezionato una cartella con 'Sfoglia...'.")
        risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
        risultato_text.insert(tk.END, "Errore: Cartella o file non validi."); risultato_text.config(state=tk.NORMAL)
        return

    risultati_globali,info_ricerca_globale = [],{}
    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END,f"Ricerca {modalita} in corso...\nAttendere prego.\n")
    risultato_text.config(state=tk.DISABLED)
    root.update_idletasks()

    aggiorna_risultati_globali([],{},modalita=modalita)

    try:
        start_dt,end_dt = start_date_entry.get_date(),end_date_entry.get_date()
        if start_dt > end_dt: raise ValueError("Data inizio dopo data fine.")
        start_ts,end_ts = pd.Timestamp(start_dt),pd.Timestamp(end_dt)
    except Exception as e:
        messagebox.showerror("Input Date",f"Date non valide: {e}")
        risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,"Errore input date."); risultato_text.config(state=tk.NORMAL)
        return

    messaggi_out,ris_graf_loc = [],[]
    col_num_nomi = [f'Numero{i+1}' for i in range(5)]

    if modalita == "successivi":
        # ... (Blocco input GUI e ricerca date trigger - INVARIATO) ...
        ra_idx,rv_idx = listbox_ruote_analisi.curselection(),listbox_ruote_verifica.curselection()
        if not ra_idx or not rv_idx:
            messagebox.showwarning("Manca Input","Seleziona Ruote Analisi (Spia) e Ruote Verifica (Esiti).")
            risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input mancante.");risultato_text.config(state=tk.NORMAL);return
        nomi_ra,nomi_rv = [listbox_ruote_analisi.get(i) for i in ra_idx],[listbox_ruote_verifica.get(i) for i in rv_idx]
        tipo_spia_scelto = tipo_spia_var_global.get() if tipo_spia_var_global else "estratto"
        numeri_spia_input_raw = [e.get().strip() for e in entry_numeri_spia]
        numeri_spia_input_validi_zfill = [str(int(n_str)).zfill(2) for n_str in numeri_spia_input_raw if n_str.isdigit() and 1 <= int(n_str) <= 90]
        numeri_spia_da_usare = None; spia_display_str = ""; posizione_spia_selezionata = None
        if tipo_spia_scelto == "estratto":
            if not numeri_spia_input_validi_zfill: messagebox.showwarning("Manca Input","Nessun Numero Spia (Estratto) valido.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input mancante.");risultato_text.config(state=tk.NORMAL);return
            numeri_spia_da_usare = numeri_spia_input_validi_zfill; spia_display_str = ", ".join(numeri_spia_da_usare)
        elif tipo_spia_scelto == "estratto_posizionale":
            if not numeri_spia_input_validi_zfill or not numeri_spia_input_raw[0].strip(): messagebox.showwarning("Manca Input","Inserire il primo Numero Spia per l'analisi posizionale.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input mancante.");risultato_text.config(state=tk.NORMAL);return
            primo_numero_spia = numeri_spia_input_validi_zfill[0]; numeri_spia_da_usare = [primo_numero_spia]; posizione_scelta_str = combo_posizione_spia.get()
            if not posizione_scelta_str or "Qualsiasi" in posizione_scelta_str: tipo_spia_scelto = "estratto"; spia_display_str = primo_numero_spia
            else:
                try: posizione_spia_selezionata = int(posizione_scelta_str.split("a")[0]); assert 1 <= posizione_spia_selezionata <= 5; spia_display_str = f"{primo_numero_spia} in {posizione_spia_selezionata}a pos."
                except: messagebox.showerror("Input Invalido",f"Posizione Spia non valida.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input non valido.");risultato_text.config(state=tk.NORMAL);return
        elif tipo_spia_scelto == "ambo":
            if len(numeri_spia_input_validi_zfill) < 2: messagebox.showwarning("Manca Input","Inserire almeno 2 numeri validi per Ambo Spia.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input mancante.");risultato_text.config(state=tk.NORMAL);return
            numeri_spia_da_usare = tuple(sorted(numeri_spia_input_validi_zfill[:2])); spia_display_str = "-".join(numeri_spia_da_usare)
        else: messagebox.showerror("Errore Interno", f"Tipo spia '{tipo_spia_scelto}' non gestito.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Errore tipo spia.");risultato_text.config(state=tk.NORMAL);return
        messaggi_out.append(f"Tipo Spia Analizzata: {tipo_spia_scelto.upper().replace('_',' ')} ({spia_display_str})")
        try: n_estr=int(estrazioni_entry_succ.get()); assert 1 <= n_estr <= MAX_COLPI_GIOCO
        except: messagebox.showerror("Input Invalido",f"N. Estrazioni (1-{MAX_COLPI_GIOCO}) non valido.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input non valido.");risultato_text.config(state=tk.NORMAL);return
        info_curr={'numeri_spia_input':numeri_spia_da_usare,'tipo_spia_usato': tipo_spia_scelto,'ruote_analisi':nomi_ra,'ruote_verifica':nomi_rv,'n_estrazioni':n_estr,'start_date':start_ts,'end_date':end_ts}
        if posizione_spia_selezionata is not None: info_curr['posizione_spia_input'] = posizione_spia_selezionata
        all_date_trig=set(); messaggi_out.append("\n--- FASE 1: Ricerca Date Uscita Spia ---")
        for nome_ra_loop in nomi_ra:
            df_an=carica_dati(file_ruote.get(nome_ra_loop),start_ts,end_ts)
            if df_an is None or df_an.empty: messaggi_out.append(f"[{nome_ra_loop}] No dati An."); continue
            dates_found_this_ruota = []
            if tipo_spia_scelto == "estratto": dates_found_arr=df_an.loc[df_an[col_num_nomi].isin(numeri_spia_da_usare).any(axis=1),'Data'].unique(); dates_found_this_ruota = pd.to_datetime(dates_found_arr).tolist()
            elif tipo_spia_scelto == "estratto_posizionale" and posizione_spia_selezionata is not None:
                numero_spia_pos = numeri_spia_da_usare[0]; colonna_target_pos = f'Numero{posizione_spia_selezionata}'
                if colonna_target_pos in df_an.columns: dates_found_arr = df_an.loc[df_an[colonna_target_pos] == numero_spia_pos, 'Data'].unique(); dates_found_this_ruota = pd.to_datetime(dates_found_arr).tolist()
            elif tipo_spia_scelto == "ambo":
                spia_n1, spia_n2 = numeri_spia_da_usare[0], numeri_spia_da_usare[1]
                presente_n1 = df_an[col_num_nomi].isin([spia_n1]).any(axis=1); presente_n2 = df_an[col_num_nomi].isin([spia_n2]).any(axis=1)
                date_ambo_presente = df_an.loc[presente_n1 & presente_n2, 'Data'].unique(); dates_found_this_ruota = pd.to_datetime(date_ambo_presente).tolist()
            if dates_found_this_ruota: all_date_trig.update(dates_found_this_ruota); messaggi_out.append(f"[{nome_ra_loop}] Trovate {len(dates_found_this_ruota)} date trigger per spia {spia_display_str}.")
            else: messaggi_out.append(f"[{nome_ra_loop}] Nessuna uscita spia {spia_display_str}.")

        if not all_date_trig:
            messaggi_out.append(f"\nNESSUNA USCITA SPIA TROVATA PER {spia_display_str}.")
            aggiorna_risultati_globali([],info_curr,modalita=modalita)
            final_output_no_trigger = "\n".join(messaggi_out)
            risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END); risultato_text.insert(tk.END,final_output_no_trigger); risultato_text.config(state=tk.NORMAL); risultato_text.see("1.0")
            mostra_popup_risultati_spia(info_ricerca_globale, risultati_globali); return
        date_trig_ord=sorted(list(all_date_trig)); n_trig_tot=len(date_trig_ord)
        messaggi_out.append(f"\nFASE 1 OK: {n_trig_tot} date trigger totali per spia {spia_display_str}."); info_curr['date_trigger_ordinate']=date_trig_ord

        messaggi_out.append("\n--- FASE 2: Analisi Ruote Verifica ---")
        df_cache_ver = {}
        df_cache_completi_per_ritardo = {}
        num_rv_ok = 0

        for nome_rv_loop in nomi_rv:
            df_ver_per_analisi_spia = df_cache_ver.get(nome_rv_loop)
            if df_ver_per_analisi_spia is None:
                temp_df_full = carica_dati(file_ruote.get(nome_rv_loop), start_date=start_ts, end_date=None)
                if temp_df_full is not None and not temp_df_full.empty:
                    df_ver_per_analisi_spia = temp_df_full
                    df_cache_ver[nome_rv_loop] = df_ver_per_analisi_spia

            df_ruota_completa_per_ritardo = df_cache_completi_per_ritardo.get(nome_rv_loop)
            if df_ruota_completa_per_ritardo is None:
                df_ruota_completa_per_ritardo = carica_dati(file_ruote.get(nome_rv_loop), start_date=None, end_date=None)
                if df_ruota_completa_per_ritardo is not None and not df_ruota_completa_per_ritardo.empty:
                    df_cache_completi_per_ritardo[nome_rv_loop] = df_ruota_completa_per_ritardo

            if df_ver_per_analisi_spia is None or df_ver_per_analisi_spia.empty:
                 messaggi_out.append(f"[{nome_rv_loop}] No dati Ver. per analisi spia."); continue

            res_ver, err_ver = analizza_ruota_verifica(df_ver_per_analisi_spia, date_trig_ord, n_estr, nome_rv_loop)

            if err_ver:
                messaggi_out.append(f"[{nome_rv_loop}] Errore: {err_ver}"); continue

            if res_ver:
                # ... (Blocco analisi per ruota con ritardi - INVARIATO dalla precedente correzione) ...
                ris_graf_loc.append((nome_rv_loop, spia_display_str, res_ver))
                num_rv_ok += 1
                msg_res_v = f"\n=== Risultati Verifica: {nome_rv_loop} (Base: {res_ver['totale_trigger']} trigger) ==="
                for tipo_s_out in ['estratto', 'ambo', 'terno']:
                    res_s_out = res_ver.get(tipo_s_out)
                    if res_s_out:
                        msg_res_v += f"\n--- {tipo_s_out.capitalize()} Successivi ---\n  Top 10 per Presenza (su {res_ver['totale_trigger']} casi trigger):\n"
                        if tipo_s_out in ['estratto', 'ambo']:
                            res_s_out.setdefault('ritardi_attuali', {})

                        if not res_s_out['presenza']['top'].empty:
                            for i, (item_key_originale, pres) in enumerate(res_s_out['presenza']['top'].items()):
                                item_str_out = item_key_originale
                                perc_p_out = res_s_out['presenza']['percentuali'].get(item_key_originale, 0.0)
                                freq_p_out = res_s_out['presenza']['frequenze'].get(item_key_originale, 0)
                                riga_output_base = f"    {i+1}. {item_str_out}: Pres. {pres} ({perc_p_out:.1f}%) | Freq.Tot: {freq_p_out}"
                                ritardo_attuale_str = ""

                                if df_ruota_completa_per_ritardo is not None and not df_ruota_completa_per_ritardo.empty:
                                    ritardo_val = "N/D"
                                    if tipo_s_out == 'estratto':
                                        ritardo_val = calcola_ritardo_attuale(df_ruota_completa_per_ritardo, item_str_out, "estratto", end_ts)
                                        if 'ritardi_attuali' in res_s_out:
                                            res_s_out['ritardi_attuali'][item_str_out] = ritardo_val
                                    elif tipo_s_out == 'ambo':
                                        try:
                                            ambo_tuple_per_ritardo = tuple(item_str_out.split('-'))
                                            if len(ambo_tuple_per_ritardo) == 2:
                                                ritardo_val = calcola_ritardo_attuale(df_ruota_completa_per_ritardo, ambo_tuple_per_ritardo, "ambo", end_ts)
                                            else: ritardo_val = "N/A (parse)"
                                        except Exception: ritardo_val = "N/A (err)"
                                        if 'ritardi_attuali' in res_s_out:
                                            res_s_out['ritardi_attuali'][item_str_out] = ritardo_val
                                    ritardo_attuale_str = f" | Rit.Att: {ritardo_val}"
                                else:
                                    ritardo_attuale_str = " | Rit.Att: N/D (no data)"
                                    if tipo_s_out in ['estratto', 'ambo'] and 'ritardi_attuali' in res_s_out:
                                         res_s_out['ritardi_attuali'][item_str_out] = "N/D (no data)"
                                msg_res_v += riga_output_base + ritardo_attuale_str + "\n"
                        else:
                            msg_res_v += "    Nessuno.\n"

                        msg_res_v+=f"  Top 10 per Frequenza Totale:\n"
                        if not res_s_out['frequenza']['top'].empty:
                            for i,(item,freq) in enumerate(res_s_out['frequenza']['top'].items()):
                                item_str_out = item; perc_f_out=res_s_out['frequenza']['percentuali'].get(item,0.0); pres_f_out=res_s_out['frequenza']['presenze'].get(item,0)
                                msg_res_v+=f"    {i+1}. {item_str_out}: Freq.Tot: {freq} ({perc_f_out:.1f}%) | Pres. su Trigger: {pres_f_out}\n"
                        else: msg_res_v+="    Nessuno.\n"
                        if tipo_s_out == 'ambo':
                            migliori_ambi_log_info = res_s_out.get('migliori_per_copertura_trigger')
                            if migliori_ambi_log_info and migliori_ambi_log_info['items']:
                                msg_res_v += f"  Migliori Ambi per Copertura Eventi Spia (su {migliori_ambi_log_info['totale_trigger_spia']} totali):\n"
                                for ambo_str_log, count_cop_log in migliori_ambi_log_info['items']:
                                    perc_cop_log = (count_cop_log / migliori_ambi_log_info['totale_trigger_spia'] * 100) if migliori_ambi_log_info['totale_trigger_spia'] > 0 else 0
                                    msg_res_v += f"    - Ambo {ambo_str_log}: Coperti {count_cop_log} eventi spia ({perc_cop_log:.1f}%)\n"
                            elif res_s_out: msg_res_v += "  Migliori Ambi per Copertura Eventi Spia: Nessuno con copertura significativa.\n"
                    else:
                        msg_res_v += f"\n--- {tipo_s_out.capitalize()} Successivi: Nessun risultato ---\n"
                messaggi_out.append(msg_res_v)
            messaggi_out.append("- " * 20)

        if ris_graf_loc and num_rv_ok > 0:
            # ... (Blocco RISULTATI COMBINATI (PER PUNTEGGIO) - INVARIATO dalla precedente correzione) ...
            messaggi_out.append("\n\n=== RISULTATI COMBINATI (PER PUNTEGGIO - TUTTE RUOTE VERIFICA) ===")
            info_curr['statistiche_combinate_dettagliate'] = {}
            info_curr.setdefault('ritardi_attuali_combinati', {})

            top_comb_ver = {'estratto': [], 'ambo': [], 'terno': []}
            peso_pres, peso_freq = 0.6, 0.4

            for tipo_comb in ['estratto', 'ambo', 'terno']:
                messaggi_out.append(f"\n--- Combinati: {tipo_comb.upper()} Successivi (per Punteggio) ---")
                comb_pres_dict, comb_freq_dict, has_data_comb = {}, {}, False
                for _, _, res_comb_loop in ris_graf_loc:
                    if res_comb_loop and res_comb_loop.get(tipo_comb):
                        has_data_comb = True
                        current_presenze = res_comb_loop[tipo_comb].get('full_presenze', pd.Series(dtype=int))
                        current_frequenze = res_comb_loop[tipo_comb].get('full_frequenze', pd.Series(dtype=int))
                        for item_c, count_c in current_presenze.items():
                            comb_pres_dict[item_c] = comb_pres_dict.get(item_c, 0) + count_c
                        for item_c, count_c in current_frequenze.items():
                            comb_freq_dict[item_c] = comb_freq_dict.get(item_c, 0) + count_c

                if not has_data_comb:
                    messaggi_out.append(f"    Nessun risultato combinato per {tipo_comb}.\n")
                    info_curr['ritardi_attuali_combinati'].setdefault(tipo_comb, {})
                    continue

                comb_pres_s_orig_keys = pd.Series(comb_pres_dict, dtype=int)
                comb_freq_s_orig_keys = pd.Series(comb_freq_dict, dtype=int)
                all_items_idx_comb_orig_keys = comb_pres_s_orig_keys.index.union(comb_freq_s_orig_keys.index)

                def get_sortable_key_comb(k): return tuple(map(str, k)) if isinstance(k, tuple) else str(k)
                sortable_index_list_comb = sorted(list(all_items_idx_comb_orig_keys), key=get_sortable_key_comb)
                ordered_index_comb = pd.Index(sortable_index_list_comb)

                comb_pres_s = comb_pres_s_orig_keys.reindex(ordered_index_comb, fill_value=0)
                comb_freq_s = comb_freq_s_orig_keys.reindex(ordered_index_comb, fill_value=0)

                tot_pres_ops_comb = n_trig_tot * num_rv_ok
                comb_perc_pres_s = (comb_pres_s / tot_pres_ops_comb * 100).round(2) if tot_pres_ops_comb > 0 else pd.Series(0.0, index=comb_pres_s.index, dtype=float)
                max_freq_comb = comb_freq_s.max() if not comb_freq_s.empty else 0
                comb_freq_norm_s = (comb_freq_s / max_freq_comb * 100).round(2) if max_freq_comb > 0 else pd.Series(0.0, index=comb_freq_s.index, dtype=float)

                punt_comb_s = ((peso_pres * comb_perc_pres_s) + (peso_freq * comb_freq_norm_s)).round(2).sort_values(ascending=False)
                top_punt_comb = punt_comb_s.head(10)

                info_curr['ritardi_attuali_combinati'].setdefault(tipo_comb, {})

                if not top_punt_comb.empty:
                    top_comb_ver[tipo_comb] = top_punt_comb.index.tolist()[:10]
                    stat_dett_comb = []
                    messaggi_out.append(f"  Top 10 Combinati per Punteggio:\n")
                    for i, (item_comb_key, score_comb_loop) in enumerate(top_punt_comb.items()):
                        item_str_formatted = format_ambo_terno(item_comb_key)

                        min_ritardo_comb = float('inf')
                        ritardo_valido_trovato = False
                        if tipo_comb in ['estratto', 'ambo']:
                            for nome_rv_rit_calc in nomi_rv:
                                df_ruota_storico = df_cache_completi_per_ritardo.get(nome_rv_rit_calc)
                                if df_ruota_storico is not None and not df_ruota_storico.empty:
                                    current_rit_val = "N/D"
                                    if tipo_comb == 'estratto':
                                        current_rit_val = calcola_ritardo_attuale(df_ruota_storico, item_str_formatted, "estratto", end_ts)
                                    elif tipo_comb == 'ambo':
                                        try:
                                            ambo_tuple = tuple(item_str_formatted.split('-'))
                                            if len(ambo_tuple) == 2:
                                                current_rit_val = calcola_ritardo_attuale(df_ruota_storico, ambo_tuple, "ambo", end_ts)
                                        except: pass

                                    if isinstance(current_rit_val, (int, float)):
                                        min_ritardo_comb = min(min_ritardo_comb, current_rit_val)
                                        ritardo_valido_trovato = True

                        ritardo_display_str = ""
                        ritardo_da_memorizzare = "N/A"
                        if ritardo_valido_trovato:
                            ritardo_display_str = f" | Rit.Min.Att: {int(min_ritardo_comb)}"
                            ritardo_da_memorizzare = int(min_ritardo_comb)
                        elif tipo_comb in ['estratto', 'ambo']:
                             ritardo_display_str = " | Rit.Min.Att: N/D"
                             ritardo_da_memorizzare = "N/D"

                        if tipo_comb in ['estratto', 'ambo']:
                            info_curr['ritardi_attuali_combinati'][tipo_comb][item_str_formatted] = ritardo_da_memorizzare

                        pres_avg = comb_perc_pres_s.get(item_comb_key, 0.0)
                        freq_tot = comb_freq_s.get(item_comb_key, 0)
                        messaggi_out.append(f"    {i+1}. {item_str_formatted}: Punt={score_comb_loop:.2f} (PresAvg:{pres_avg:.1f}%, FreqTot:{freq_tot}){ritardo_display_str}\n")
                        stat_dett_comb.append({
                            "item": item_str_formatted,
                            "punteggio": score_comb_loop,
                            "presenza_media_perc": pres_avg,
                            "frequenza_totale": freq_tot,
                            "ritardo_min_attuale": ritardo_da_memorizzare
                        })
                    info_curr.setdefault('statistiche_combinate_dettagliate', {})[tipo_comb] = stat_dett_comb
                else:
                    messaggi_out.append("    Nessuno.\n")
            info_curr['top_combinati'] = top_comb_ver
        elif num_rv_ok == 0: messaggi_out.append("\nNessuna Ruota Verifica valida con risultati.")

        if ris_graf_loc and num_rv_ok > 0 and n_trig_tot > 0:
            # ... (Blocco MIGLIORI AMBI PER COPERTURA GLOBALE - CON CORREZIONI F-STRING E LOGICA RITARDI) ...
            messaggi_out.append("\n\n=== MIGLIORI AMBI PER COPERTURA GLOBALE DEGLI EVENTI SPIA ===")
            messaggi_out.append(f"(Su {n_trig_tot} eventi spia totali, considerando uscite su QUALSIASI ruota di verifica)")
            ambi_copertura_globale_eventi = {}
            for data_trigger_evento in date_trig_ord:
                ambi_usciti_per_questo_trigger_globale = set()
                for nome_rv_check_glob in nomi_rv:
                    df_ver_loop = df_cache_completi_per_ritardo.get(nome_rv_check_glob)
                    if df_ver_loop is None or df_ver_loop.empty: continue
                    date_series_ver_loop = df_ver_loop['Data']
                    try: start_index_loop = date_series_ver_loop.searchsorted(data_trigger_evento, side='right')
                    except Exception: continue
                    if start_index_loop >= len(date_series_ver_loop): continue
                    df_successive_loop = df_ver_loop.iloc[start_index_loop : start_index_loop + n_estr]
                    if not df_successive_loop.empty:
                        for _, row_loop in df_successive_loop.iterrows():
                            numeri_riga_loop = sorted([str(row_loop[col]).zfill(2) for col in col_num_nomi if pd.notna(row_loop[col])])
                            if len(numeri_riga_loop) >= 2:
                                for ambo_tuple_loop in itertools.combinations(numeri_riga_loop, 2): ambi_usciti_per_questo_trigger_globale.add(ambo_tuple_loop)
                for ambo_coperto_glob in ambi_usciti_per_questo_trigger_globale:
                    if ambo_coperto_glob not in ambi_copertura_globale_eventi: ambi_copertura_globale_eventi[ambo_coperto_glob] = set()
                    ambi_copertura_globale_eventi[ambo_coperto_glob].add(data_trigger_evento)
            conteggio_copertura_globale = Counter({ambo_glob: len(date_coperte_glob) for ambo_glob, date_coperte_glob in ambi_copertura_globale_eventi.items()})

            if not conteggio_copertura_globale:
                messaggi_out.append("    Nessun ambo ha coperto eventi spia (considerando tutte le ruote di verifica).")
                info_curr['migliori_ambi_copertura_globale'] = []
            else:
                migliori_ambi_globali_raw = sorted(conteggio_copertura_globale.items(), key=lambda item: (item[1], item[0]), reverse=True)
                num_top_ambi_globali_display = min(len(migliori_ambi_globali_raw), 10)
                info_curr['migliori_ambi_copertura_globale'] = []

                if num_top_ambi_globali_display > 0 :
                    for i in range(num_top_ambi_globali_display):
                        ambo_glob_tuple, count_glob = migliori_ambi_globali_raw[i]
                        ambo_glob_str = format_ambo_terno(ambo_glob_tuple)
                        perc_glob = (count_glob / n_trig_tot * 100) if n_trig_tot > 0 else 0

                        min_rit_ambo_glob = float('inf')
                        rit_valido_ambo_glob_trovato = False
                        for nome_rv_rit_calc_glob in nomi_rv:
                            df_ruota_storico_glob = df_cache_completi_per_ritardo.get(nome_rv_rit_calc_glob)
                            if df_ruota_storico_glob is not None and not df_ruota_storico_glob.empty:
                                current_rit_val_glob = calcola_ritardo_attuale(df_ruota_storico_glob, ambo_glob_tuple, "ambo", end_ts)
                                if isinstance(current_rit_val_glob, (int, float)):
                                    min_rit_ambo_glob = min(min_rit_ambo_glob, current_rit_val_glob)
                                    rit_valido_ambo_glob_trovato = True

                        rit_display_ambo_glob_str = ""
                        rit_da_mem_ambo_glob = "N/A"
                        if rit_valido_ambo_glob_trovato:
                            rit_display_ambo_glob_str = f" | Rit.Min.Att: {int(min_rit_ambo_glob)}"
                            rit_da_mem_ambo_glob = int(min_rit_ambo_glob)
                        else:
                            rit_display_ambo_glob_str = " | Rit.Min.Att: N/D"
                            rit_da_mem_ambo_glob = "N/D"

                        messaggi_out.append(f"    {i+1}. Ambo {ambo_glob_str}: Coperti {count_glob} su {n_trig_tot} eventi spia ({perc_glob:.1f}%){rit_display_ambo_glob_str}")
                        info_curr['migliori_ambi_copertura_globale'].append({
                            "ambo": ambo_glob_str, "coperti": count_glob, "totali": n_trig_tot,
                            "percentuale": perc_glob, "ritardo_min_attuale": rit_da_mem_ambo_glob
                        })

                    info_curr['combinazione_ottimale_copertura_100'] = None
                    info_curr['migliore_combinazione_parziale'] = None
                    if len(info_curr['migliori_ambi_copertura_globale']) > 0 and n_trig_tot > 0:
                        messaggi_out.append(f"\n  RICERCA COMBINAZIONE AMBI PER COPERTURA TOTALE ({n_trig_tot} eventi spia):")
                        soluzione_trovata = False

                        ambo_top1_info_comb = info_curr['migliori_ambi_copertura_globale'][0]
                        rit_top1_val_comb = ambo_top1_info_comb['ritardo_min_attuale']
                        rit_top1_display_comb = ""
                        if rit_top1_val_comb not in ['N/A', 'N/D', None]: rit_top1_display_comb = f" [Rit.Min.Att: {rit_top1_val_comb}]"
                        elif rit_top1_val_comb == 'N/D': rit_top1_display_comb = " [Rit.Min.Att: N/D]"
                        ambo_top1_str_det_comb = f"{ambo_top1_info_comb['ambo']}{rit_top1_display_comb}"

                        ambo_top1_tuple_raw_comb = tuple(ambo_top1_info_comb["ambo"].split('-'))
                        date_coperte_top1_comb = ambi_copertura_globale_eventi.get(ambo_top1_tuple_raw_comb, set())

                        if len(date_coperte_top1_comb) == n_trig_tot:
                            messaggi_out.append(f"  - L'ambo singolo '{ambo_top1_str_det_comb}' copre il 100% ({n_trig_tot}/{n_trig_tot} eventi).")
                            info_curr['combinazione_ottimale_copertura_100'] = {"ambi_dettagli": [ambo_top1_str_det_comb], "ambi": [ambo_top1_info_comb['ambo']], "coperti": n_trig_tot, "totali": n_trig_tot, "percentuale": 100.0}
                            soluzione_trovata = True

                        if not soluzione_trovata and len(info_curr['migliori_ambi_copertura_globale']) >= 2:
                            miglior_coppia_copertura = 0; miglior_coppia_ambi_dettagli_list = []
                            for combo_2_idx in itertools.combinations(range(min(len(info_curr['migliori_ambi_copertura_globale']), 7)), 2):
                                amboA_info_comb2 = info_curr['migliori_ambi_copertura_globale'][combo_2_idx[0]]
                                amboB_info_comb2 = info_curr['migliori_ambi_copertura_globale'][combo_2_idx[1]]
                                rit_A_val_comb2 = amboA_info_comb2['ritardo_min_attuale']; rit_A_display_comb2 = ""
                                if rit_A_val_comb2 not in ['N/A', 'N/D', None]: rit_A_display_comb2 = f" [Rit.Min.Att: {rit_A_val_comb2}]"
                                elif rit_A_val_comb2 == 'N/D': rit_A_display_comb2 = " [Rit.Min.Att: N/D]"
                                rit_B_val_comb2 = amboB_info_comb2['ritardo_min_attuale']; rit_B_display_comb2 = ""
                                if rit_B_val_comb2 not in ['N/A', 'N/D', None]: rit_B_display_comb2 = f" [Rit.Min.Att: {rit_B_val_comb2}]"
                                elif rit_B_val_comb2 == 'N/D': rit_B_display_comb2 = " [Rit.Min.Att: N/D]"
                                current_ambi_dettagli_comb_list = sorted([f"{amboA_info_comb2['ambo']}{rit_A_display_comb2}", f"{amboB_info_comb2['ambo']}{rit_B_display_comb2}"])

                                amboA_tuple_comb_raw2 = tuple(amboA_info_comb2["ambo"].split('-')); amboB_tuple_comb_raw2 = tuple(amboB_info_comb2["ambo"].split('-'))
                                date_A_comb2 = ambi_copertura_globale_eventi.get(amboA_tuple_comb_raw2, set()); date_B_comb2 = ambi_copertura_globale_eventi.get(amboB_tuple_comb_raw2, set())
                                coperte_da_coppia_comb2 = date_A_comb2.union(date_B_comb2)

                                if len(coperte_da_coppia_comb2) == n_trig_tot:
                                    miglior_coppia_ambi_dettagli_list = current_ambi_dettagli_comb_list
                                    messaggi_out.append(f"  - La coppia '{miglior_coppia_ambi_dettagli_list[0]}' e '{miglior_coppia_ambi_dettagli_list[1]}' copre il 100% ({n_trig_tot}/{n_trig_tot} eventi).")
                                    info_curr['combinazione_ottimale_copertura_100'] = {"ambi_dettagli": miglior_coppia_ambi_dettagli_list, "ambi": [info["ambo"] for info in [amboA_info_comb2, amboB_info_comb2]], "coperti": n_trig_tot, "totali": n_trig_tot, "percentuale": 100.0}
                                    soluzione_trovata = True; break
                                if len(coperte_da_coppia_comb2) > miglior_coppia_copertura:
                                    miglior_coppia_copertura = len(coperte_da_coppia_comb2); miglior_coppia_ambi_dettagli_list = current_ambi_dettagli_comb_list
                            if not soluzione_trovata and miglior_coppia_ambi_dettagli_list:
                                info_curr['migliore_combinazione_parziale'] = {"ambi_dettagli": miglior_coppia_ambi_dettagli_list, "ambi": [a.split(' ')[0] for a in miglior_coppia_ambi_dettagli_list], "coperti": miglior_coppia_copertura, "totali": n_trig_tot, "percentuale": (miglior_coppia_copertura/n_trig_tot*100) if n_trig_tot > 0 else 0}

                        if not soluzione_trovata and len(info_curr['migliori_ambi_copertura_globale']) >= 3:
                            miglior_terzina_copertura = 0; miglior_terzina_ambi_dettagli_list = []
                            for combo_3_idx in itertools.combinations(range(min(len(info_curr['migliori_ambi_copertura_globale']), 7)), 3):
                                amboA_info_comb3 = info_curr['migliori_ambi_copertura_globale'][combo_3_idx[0]]; amboB_info_comb3 = info_curr['migliori_ambi_copertura_globale'][combo_3_idx[1]]; amboC_info_comb3 = info_curr['migliori_ambi_copertura_globale'][combo_3_idx[2]]
                                rit_A_val_comb3 = amboA_info_comb3['ritardo_min_attuale']; rit_A_display_comb3 = ""
                                if rit_A_val_comb3 not in ['N/A', 'N/D', None]: rit_A_display_comb3 = f" [Rit.Min.Att: {rit_A_val_comb3}]"
                                elif rit_A_val_comb3 == 'N/D': rit_A_display_comb3 = " [Rit.Min.Att: N/D]"
                                rit_B_val_comb3 = amboB_info_comb3['ritardo_min_attuale']; rit_B_display_comb3 = ""
                                if rit_B_val_comb3 not in ['N/A', 'N/D', None]: rit_B_display_comb3 = f" [Rit.Min.Att: {rit_B_val_comb3}]"
                                elif rit_B_val_comb3 == 'N/D': rit_B_display_comb3 = " [Rit.Min.Att: N/D]"
                                rit_C_val_comb3 = amboC_info_comb3['ritardo_min_attuale']; rit_C_display_comb3 = ""
                                if rit_C_val_comb3 not in ['N/A', 'N/D', None]: rit_C_display_comb3 = f" [Rit.Min.Att: {rit_C_val_comb3}]"
                                elif rit_C_val_comb3 == 'N/D': rit_C_display_comb3 = " [Rit.Min.Att: N/D]"
                                current_ambi_dettagli_comb3_list = sorted([f"{amboA_info_comb3['ambo']}{rit_A_display_comb3}", f"{amboB_info_comb3['ambo']}{rit_B_display_comb3}", f"{amboC_info_comb3['ambo']}{rit_C_display_comb3}"])

                                amboA_tuple_comb3_raw = tuple(amboA_info_comb3["ambo"].split('-')); amboB_tuple_comb3_raw = tuple(amboB_info_comb3["ambo"].split('-')); amboC_tuple_comb3_raw = tuple(amboC_info_comb3["ambo"].split('-'))
                                dcA_comb3 = ambi_copertura_globale_eventi.get(amboA_tuple_comb3_raw, set()); dcB_comb3 = ambi_copertura_globale_eventi.get(amboB_tuple_comb3_raw, set()); dcC_comb3 = ambi_copertura_globale_eventi.get(amboC_tuple_comb3_raw, set())
                                coperte_da_terzina_comb3 = dcA_comb3.union(dcB_comb3).union(dcC_comb3)
                                if len(coperte_da_terzina_comb3) == n_trig_tot:
                                    miglior_terzina_ambi_dettagli_list = current_ambi_dettagli_comb3_list
                                    messaggi_out.append(f"  - La terzina '{miglior_terzina_ambi_dettagli_list[0]}', '{miglior_terzina_ambi_dettagli_list[1]}' e '{miglior_terzina_ambi_dettagli_list[2]}' copre il 100% ({n_trig_tot}/{n_trig_tot} eventi).")
                                    info_curr['combinazione_ottimale_copertura_100'] = {"ambi_dettagli": miglior_terzina_ambi_dettagli_list, "ambi": [info["ambo"] for info in [amboA_info_comb3, amboB_info_comb3, amboC_info_comb3]], "coperti": n_trig_tot, "totali": n_trig_tot, "percentuale": 100.0}
                                    soluzione_trovata = True; break
                                if len(coperte_da_terzina_comb3) > miglior_terzina_copertura:
                                    miglior_terzina_copertura = len(coperte_da_terzina_comb3); miglior_terzina_ambi_dettagli_list = current_ambi_dettagli_comb3_list
                            if not soluzione_trovata and miglior_terzina_ambi_dettagli_list and (not info_curr.get('migliore_combinazione_parziale') or miglior_terzina_copertura > info_curr['migliore_combinazione_parziale']['coperti']):
                                info_curr['migliore_combinazione_parziale'] = {"ambi_dettagli": miglior_terzina_ambi_dettagli_list, "ambi": [a.split(' ')[0] for a in miglior_terzina_ambi_dettagli_list], "coperti": miglior_terzina_copertura, "totali": n_trig_tot, "percentuale": (miglior_terzina_copertura/n_trig_tot*100) if n_trig_tot > 0 else 0}

                        if not soluzione_trovata:
                            messaggi_out.append("  - Non è stata trovata una combinazione di 1, 2 o 3 ambi (dai top considerati) per la copertura del 100%.")
                            if info_curr['migliore_combinazione_parziale']:
                                parziale = info_curr['migliore_combinazione_parziale']
                                ambi_da_mostrare_log_parziale = parziale.get("ambi_dettagli", parziale["ambi"])
                                messaggi_out.append(f"    La migliore copertura parziale trovata con {len(parziale['ambi'])} ambo/i ({', '.join(ambi_da_mostrare_log_parziale)}) è di {parziale['coperti']}/{parziale['totali']} eventi ({parziale['percentuale']:.1f}%).")
                else:
                     messaggi_out.append("    Nessun ambo ha coperto eventi spia in modo significativo (considerando tutte le ruote di verifica).")

        aggiorna_risultati_globali(ris_graf_loc,info_curr,modalita="successivi")
        if ris_graf_loc or (not all_date_trig and tipo_spia_scelto):
            mostra_popup_risultati_spia(info_ricerca_globale, risultati_globali)

    elif modalita == "antecedenti":
        # ... (codice modalità antecedenti - INVARIATO) ...
        # (come nella tua versione precedente)
        ra_ant_idx=listbox_ruote_analisi_ant.curselection()
        if not ra_ant_idx: messagebox.showwarning("Manca Input","Seleziona Ruota/e Analisi.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input mancante.");risultato_text.config(state=tk.NORMAL);return
        nomi_ra_ant=[listbox_ruote_analisi_ant.get(i) for i in ra_ant_idx]
        num_obj_raw = [e.get().strip() for e in entry_numeri_obiettivo if e.get().strip() and e.get().strip().isdigit() and 1<=int(e.get().strip())<=90]
        num_obj = sorted(list(set(str(int(n)).zfill(2) for n in num_obj_raw)))
        if not num_obj: messagebox.showwarning("Manca Input","Numeri Obiettivo non validi.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input mancante.");risultato_text.config(state=tk.NORMAL);return
        try: n_prec=int(estrazioni_entry_ant.get()); assert n_prec >=1
        except: messagebox.showerror("Input Invalido","N. Precedenti (>=1) non valido.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END,"Input non valido.");risultato_text.config(state=tk.NORMAL);return
        messaggi_out.append(f"--- Analisi Antecedenti (Marker) ---"); messaggi_out.append(f"Numeri Obiettivo: {', '.join(num_obj)}"); messaggi_out.append(f"Numero Estrazioni Precedenti Controllate: {n_prec}"); messaggi_out.append(f"Periodo: {start_ts.strftime('%d/%m/%Y')} - {end_ts.strftime('%d/%m/%Y')}"); messaggi_out.append("-" * 40)
        df_cache_ant={}; almeno_un_risultato_antecedente = False
        for nome_ra_ant_loop in nomi_ra_ant:
            df_ant_full = df_cache_ant.get(nome_ra_ant_loop)
            if df_ant_full is None: df_ant_full=carica_dati(file_ruote.get(nome_ra_ant_loop),start_ts,end_ts); df_cache_ant[nome_ra_ant_loop]=df_ant_full
            if df_ant_full is None or df_ant_full.empty: messaggi_out.append(f"\n[{nome_ra_ant_loop.upper()}] Nessun dato storico."); continue
            res_ant,err_ant=analizza_antecedenti(df_ruota=df_ant_full, numeri_obiettivo=num_obj, n_precedenti=n_prec, nome_ruota=nome_ra_ant_loop)
            if err_ant: messaggi_out.append(f"\n[{nome_ra_ant_loop.upper()}] Errore: {err_ant}"); continue
            if res_ant and res_ant.get('base_presenza_antecedenti',0)>0 and ((res_ant.get('presenza') and not res_ant['presenza']['top'].empty) or (res_ant.get('frequenza') and not res_ant['frequenza']['top'].empty) ):
                almeno_un_risultato_antecedente = True
                msg_res_ant=f"\n=== Risultati Antecedenti per Ruota: {nome_ra_ant_loop.upper()} ==="
                msg_res_ant+=f"\n(Obiettivi: {', '.join(res_ant['numeri_obiettivo'])} | Estrazioni Prec.: {res_ant['n_precedenti']} | Occorrenze Obiettivo: {res_ant['totale_occorrenze_obiettivo']})"
                if res_ant.get('presenza') and not res_ant['presenza']['top'].empty:
                    msg_res_ant+=f"\n  Top Antecedenti per Presenza (su {res_ant['base_presenza_antecedenti']} casi validi):"
                    for i,(num,pres) in enumerate(res_ant['presenza']['top'].head(10).items()): perc_pres_val = res_ant['presenza']['percentuali'].get(num,0.0); freq_val = res_ant['presenza']['frequenze'].get(num,0); msg_res_ant+=f"\n    {i+1}. {str(num).zfill(2)}: {pres} ({perc_pres_val:.1f}%) [Freq.Tot: {freq_val}]"
                else: msg_res_ant+="\n  Nessun Top per Presenza."
                if res_ant.get('frequenza') and not res_ant['frequenza']['top'].empty:
                    msg_res_ant+=f"\n  Top Antecedenti per Frequenza Totale:"
                    for i,(num,freq) in enumerate(res_ant['frequenza']['top'].head(10).items()): perc_freq_val = res_ant['frequenza']['percentuali'].get(num,0.0); pres_val = res_ant['frequenza']['presenze'].get(num,0); msg_res_ant+=f"\n    {i+1}. {str(num).zfill(2)}: {freq} ({perc_freq_val:.1f}%) [Pres. su Casi: {pres_val}]"
                else: msg_res_ant+="\n  Nessun Top per Frequenza."
                messaggi_out.append(msg_res_ant)
            else: messaggi_out.append(f"\n[{nome_ra_ant_loop.upper()}] Nessun dato antecedente significativo.")
            messaggi_out.append("\n" + ("- "*20))
        aggiorna_risultati_globali([],{},modalita="antecedenti")

    final_output="\n".join(messaggi_out) if messaggi_out else "Nessun risultato."
    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END,final_output)
    risultato_text.see("1.0")

    if modalita == "antecedenti" and almeno_un_risultato_antecedente:
        mostra_popup_testo_semplice("Riepilogo Analisi Numeri Antecedenti", final_output)

# FUNZIONI PER VERIFICA ESITI (Logica "IN CORSO" già presente)
# =============================================================================
def verifica_esiti_utente_su_triggers(date_triggers, combinazioni_utente, nomi_ruote_verifica, n_verifiche, start_ts, end_ts, titolo_sezione="VERIFICA MISTA SU TRIGGER"):
    if not date_triggers or not combinazioni_utente or not nomi_ruote_verifica:
        return "Errore: Dati input mancanti per verifica utente su triggers."
    estratti_u, ambi_u_tpl, terni_u_tpl, quaterne_u_tpl, cinquine_u_tpl = [], [], [], [], []
    if isinstance(combinazioni_utente, dict):
        estratti_u = sorted(list(set(combinazioni_utente.get('estratto', []))))
        ambi_u_tpl = sorted(list(set(tuple(sorted(a)) for a in combinazioni_utente.get('ambo', []) if isinstance(a, (list, tuple)) and len(a) == 2)))
        terni_u_tpl = sorted(list(set(tuple(sorted(t)) for t in combinazioni_utente.get('terno', []) if isinstance(t, (list, tuple)) and len(t) == 3)))
        quaterne_u_tpl = sorted(list(set(tuple(sorted(q)) for q in combinazioni_utente.get('quaterna', []) if isinstance(q, (list, tuple)) and len(q) == 4)))
        cinquine_u_tpl = sorted(list(set(tuple(sorted(c)) for c in combinazioni_utente.get('cinquina', []) if isinstance(c, (list, tuple)) and len(c) == 5)))

    cols_num = [f'Numero{i+1}' for i in range(5)]; df_cache_ver = {}; ruote_valide = []
    for nome_rv_loop in nomi_ruote_verifica:
        df_ver = carica_dati(file_ruote.get(nome_rv_loop), start_date=start_ts, end_date=None)
        if df_ver is not None and not df_ver.empty:
            df_cache_ver[nome_rv_loop] = df_ver.sort_values(by='Data').drop_duplicates(subset=['Data']).reset_index(drop=True); ruote_valide.append(nome_rv_loop)

    if not ruote_valide: return "Errore: Nessuna ruota di verifica valida per caricare i dati del periodo."

    sorted_date_triggers = sorted(list(date_triggers)); num_casi_trigger_base = len(sorted_date_triggers)
    if num_casi_trigger_base == 0: return "Nessun caso trigger da verificare."

    out = [f"\n\n=== {titolo_sezione} ({n_verifiche} Colpi dopo ogni Trigger) ==="]
    out.append(f"Numero di casi trigger di base: {num_casi_trigger_base}"); out.append(f"Ruote di verifica considerate: {', '.join(ruote_valide) or 'Nessuna'}")

    esiti_dettagliati_per_item = {}; items_config_da_verificare = []
    if estratti_u: items_config_da_verificare.extend([('estratto', e) for e in estratti_u])
    if ambi_u_tpl: items_config_da_verificare.extend([('ambo', a) for a in ambi_u_tpl])
    if terni_u_tpl: items_config_da_verificare.extend([('terno', t) for t in terni_u_tpl])
    if quaterne_u_tpl: items_config_da_verificare.extend([('quaterna', q) for q in quaterne_u_tpl])
    if cinquine_u_tpl: items_config_da_verificare.extend([('cinquina', c) for c in cinquine_u_tpl])

    sfaldamenti_totali_per_item_ruota = {}

    for tipo_sorte_item, item_val in items_config_da_verificare:
        item_str_key = format_ambo_terno(item_val) if isinstance(item_val, tuple) else item_val
        esiti_dettagliati_per_item[item_str_key] = []
        num_eventi_trigger_coperti_per_item = 0
        total_actual_hits_for_item = 0 # NUOVO: Contatore per le uscite totali dell'item

        sfaldamenti_totali_per_item_ruota[item_str_key] = Counter()

        for data_t in sorted_date_triggers:
            dettagli_uscita_per_questo_trigger_lista = []
            trigger_coperto_in_questo_ciclo_per_item = False
            max_colpi_effettivi_per_trigger = 0
            ruote_con_hit_gia_contate_per_questo_trigger = set()

            for nome_rv in ruote_valide:
                df_v = df_cache_ver.get(nome_rv);
                if df_v is None: continue
                date_s_v = df_v['Data']
                try: start_idx = date_s_v.searchsorted(data_t, side='right')
                except Exception: continue
                if start_idx >= len(date_s_v): continue

                df_fin_v = df_v.iloc[start_idx : start_idx + n_verifiche];
                max_colpi_effettivi_per_trigger = max(max_colpi_effettivi_per_trigger, len(df_fin_v))

                if not df_fin_v.empty:
                    for colpo_idx, (_, row) in enumerate(df_fin_v.iterrows(), 1):
                        data_estrazione_corrente = row['Data'].date();
                        current_row_numbers_v = [row[col] for col in cols_num if pd.notna(row[col])];
                        nums_draw_set = set(current_row_numbers_v)
                        match = False
                        if tipo_sorte_item == 'estratto':
                            if item_val in nums_draw_set: match = True
                        elif tipo_sorte_item == 'ambo':
                            if isinstance(item_val, tuple) and set(item_val).issubset(nums_draw_set): match = True
                        elif tipo_sorte_item == 'terno':
                             if isinstance(item_val, tuple) and set(item_val).issubset(nums_draw_set): match = True
                        elif tipo_sorte_item == 'quaterna':
                             if isinstance(item_val, tuple) and set(item_val).issubset(nums_draw_set): match = True
                        elif tipo_sorte_item == 'cinquina':
                             if isinstance(item_val, tuple) and set(item_val).issubset(nums_draw_set): match = True

                        if match:
                            total_actual_hits_for_item += 1 # Incrementa il conteggio totale delle uscite
                            dettagli_uscita_per_questo_trigger_lista.append(f"{nome_rv} @ C{colpo_idx} ({data_estrazione_corrente.strftime('%d/%m')})")
                            trigger_coperto_in_questo_ciclo_per_item = True
                            
                            if nome_rv not in ruote_con_hit_gia_contate_per_questo_trigger:
                               sfaldamenti_totali_per_item_ruota[item_str_key][nome_rv] += 1
                               ruote_con_hit_gia_contate_per_questo_trigger.add(nome_rv)
                            break 
            
            esito_per_questo_trigger_str = ""
            if trigger_coperto_in_questo_ciclo_per_item:
                esito_per_questo_trigger_str = "[" + "; ".join(dettagli_uscita_per_questo_trigger_lista) + "]"
                num_eventi_trigger_coperti_per_item +=1
            elif max_colpi_effettivi_per_trigger < n_verifiche:
                esito_per_questo_trigger_str = f"IN CORSO (max {max_colpi_effettivi_per_trigger}/{n_verifiche} colpi analizzabili)"
            else:
                esito_per_questo_trigger_str = "NON USCITO"
            
            esiti_dettagliati_per_item[item_str_key].append(f"    {data_t.strftime('%d/%m/%y')}: {esito_per_questo_trigger_str}")

        out.append(f"\n--- Esiti {tipo_sorte_item.upper()} ---");
        out.append(f"  - {item_str_key}:");
        out.extend(esiti_dettagliati_per_item[item_str_key])
        
        out.append(f"\n    RIEPILOGO SFALDAMENTI SU RUOTA per {item_str_key.upper()}:")
        if not ruote_valide:
             out.append("      Nessuna ruota di verifica attiva.")
        else:
            for nome_rv_riepilogo in ruote_valide:
                conteggio = sfaldamenti_totali_per_item_ruota[item_str_key].get(nome_rv_riepilogo, 0)
                out.append(f"      - {nome_rv_riepilogo}: {conteggio} volt{'a' if conteggio == 1 else 'e'}")
        
        if num_casi_trigger_base == 0:
            out.append(f"    RIEPILOGO {item_str_key}: Nessun evento spia registrato per l'analisi.")
        elif total_actual_hits_for_item == 0:
            perc_copertura_eventi_spia = 0.0
            out.append(f"    RIEPILOGO {item_str_key}: Uscito {total_actual_hits_for_item} volte su {num_casi_trigger_base} eventi spia ({perc_copertura_eventi_spia:.1f}%)")
        else:
            perc_copertura_eventi_spia = (num_eventi_trigger_coperti_per_item / num_casi_trigger_base * 100) if num_casi_trigger_base > 0 else 0
            out.append(f"    RIEPILOGO {item_str_key}: Uscito {total_actual_hits_for_item} volte su {num_casi_trigger_base} eventi spia ({perc_copertura_eventi_spia:.1f}%)")
            
    return "\n".join(out)

def verifica_esiti_futuri(top_combinati_input, nomi_ruote_verifica, data_fine_analisi, n_colpi_futuri):
    if not top_combinati_input or not any(top_combinati_input.values()) or not nomi_ruote_verifica or data_fine_analisi is None or n_colpi_futuri <= 0:
        return "Errore: Input invalidi per verifica_esiti_futuri (post-analisi)."
    estratti_items = sorted(list(set(top_combinati_input.get('estratto', []))))
    ambi_items = sorted(list(set(tuple(sorted(a)) for a in top_combinati_input.get('ambo', []) if isinstance(a, (list, tuple)) and len(a) == 2)))
    terni_items = sorted(list(set(tuple(sorted(t)) for t in top_combinati_input.get('terno', []) if isinstance(t, (list, tuple)) and len(t) == 3)))
    quaterne_items = sorted(list(set(tuple(sorted(q)) for q in top_combinati_input.get('quaterna', []) if isinstance(q, (list, tuple)) and len(q) == 4)))
    cinquine_items = sorted(list(set(tuple(sorted(c)) for c in top_combinati_input.get('cinquina', []) if isinstance(c, (list, tuple)) and len(c) == 5)))
    cols_num = [f'Numero{i+1}' for i in range(5)]; df_cache_ver_fut = {}; ruote_con_dati_fut = []
    for nome_rv in nomi_ruote_verifica:
        df_ver_full = carica_dati(file_ruote.get(nome_rv), start_date=None, end_date=None)
        if df_ver_full is None or df_ver_full.empty: continue
        df_ver_fut_ruota = df_ver_full[df_ver_full['Data'] > data_fine_analisi].copy().sort_values(by='Data').reset_index(drop=True)
        df_fin_fut_ruota = df_ver_fut_ruota.head(n_colpi_futuri)
        if not df_fin_fut_ruota.empty: df_cache_ver_fut[nome_rv] = df_fin_fut_ruota; ruote_con_dati_fut.append(nome_rv)
    if not ruote_con_dati_fut: return f"Nessuna estrazione trovata su nessuna ruota di verifica dopo {data_fine_analisi.strftime('%d/%m/%Y')} per {n_colpi_futuri} colpi."
    num_colpi_globalmente_analizzabili = n_colpi_futuri
    if ruote_con_dati_fut: min_len = float('inf');_=[(min_len:=min(min_len,len(df_cache_ver_fut.get(r,pd.DataFrame())))) for r in ruote_con_dati_fut];num_colpi_globalmente_analizzabili=min_len
    else: num_colpi_globalmente_analizzabili = 0
    hits_registrati = {'estratto':{e:None for e in estratti_items},'ambo':{a:None for a in ambi_items},'terno':{t:None for t in terni_items},'quaterna':{q:None for q in quaterne_items},'cinquina':{c:None for c in cinquine_items}}
    for nome_rv in ruote_con_dati_fut:
        df_finestra_ruota = df_cache_ver_fut[nome_rv]
        for colpo_idx, (_, row) in enumerate(df_finestra_ruota.iterrows(), 1):
            data_estrazione_corrente = row['Data'].date(); numeri_estratti_riga = [row[col] for col in cols_num if pd.notna(row[col])]; numeri_sortati_riga = sorted(numeri_estratti_riga)
            if not numeri_sortati_riga: continue
            set_numeri_riga_formattati = set(numeri_sortati_riga)
            if estratti_items:
                for item_e_orig in estratti_items:
                    if hits_registrati['estratto'][item_e_orig] is None and format_ambo_terno(item_e_orig) in set_numeri_riga_formattati: hits_registrati['estratto'][item_e_orig] = (nome_rv, colpo_idx, data_estrazione_corrente)
            if ambi_items and len(numeri_sortati_riga) >= 2:
                for item_a_orig in ambi_items:
                    if hits_registrati['ambo'][item_a_orig] is None and set(item_a_orig).issubset(set_numeri_riga_formattati): hits_registrati['ambo'][item_a_orig] = (nome_rv, colpo_idx, data_estrazione_corrente)
            if terni_items and len(numeri_sortati_riga) >= 3:
                for item_t_orig in terni_items:
                    if hits_registrati['terno'][item_t_orig] is None and set(item_t_orig).issubset(set_numeri_riga_formattati): hits_registrati['terno'][item_t_orig] = (nome_rv, colpo_idx, data_estrazione_corrente)
            if quaterne_items and len(numeri_sortati_riga) >= 4:
                for item_q_orig in quaterne_items:
                     if hits_registrati['quaterna'][item_q_orig] is None and set(item_q_orig).issubset(set_numeri_riga_formattati): hits_registrati['quaterna'][item_q_orig] = (nome_rv, colpo_idx, data_estrazione_corrente)
            if cinquine_items and len(numeri_sortati_riga) >= 5:
                for item_c_orig in cinquine_items:
                    if hits_registrati['cinquina'][item_c_orig] is None and set(item_c_orig).issubset(set_numeri_riga_formattati): hits_registrati['cinquina'][item_c_orig] = (nome_rv, colpo_idx, data_estrazione_corrente)
    out = [f"\n\n=== VERIFICA ESITI FUTURI (POST-ANALISI) ({n_colpi_futuri} Colpi dopo {data_fine_analisi.strftime('%d/%m/%Y')}) ==="]
    out.append(f"Ruote verificate con dati futuri disponibili: {', '.join(ruote_con_dati_fut) or 'Nessuna'}")
    if ruote_con_dati_fut: out.append(f"(Analisi basata su un minimo di {num_colpi_globalmente_analizzabili} colpi disponibili globalmente su queste ruote)")
    sorti_config = [('estratto', estratti_items), ('ambo', ambi_items), ('terno', terni_items),('quaterna', quaterne_items), ('cinquina', cinquine_items)]
    for tipo_sorte, lista_items_sorte in sorti_config:
        if not lista_items_sorte: continue
        out.append(f"\n--- Esiti Futuri {tipo_sorte.upper()} ---"); almeno_un_hit_per_sorte = False
        for item_da_verificare_orig in lista_items_sorte:
            item_str_formattato = format_ambo_terno(item_da_verificare_orig); dettaglio_hit = hits_registrati[tipo_sorte].get(item_da_verificare_orig)
            if dettaglio_hit: almeno_un_hit_per_sorte = True; d_ruota, d_colpo, d_data = dettaglio_hit; out.append(f"    - {item_str_formattato}: USCITO -> {d_ruota} @ C{d_colpo} ({d_data.strftime('%d/%m/%Y')})")
            else:
                if num_colpi_globalmente_analizzabili < n_colpi_futuri and ruote_con_dati_fut : out.append(f"    - {item_str_formattato}: IN CORSO (analizzati min {num_colpi_globalmente_analizzabili}/{n_colpi_futuri} colpi)")
                else: out.append(f"    - {item_str_formattato}: NON uscito")
        if not almeno_un_hit_per_sorte and lista_items_sorte and not (num_colpi_globalmente_analizzabili < n_colpi_futuri and ruote_con_dati_fut): out.append(f"    Nessuno degli elementi {tipo_sorte.upper()} è uscito nei colpi futuri analizzati.")
    return "\n".join(out)


# MODIFICATO per gestione stato risultato_text per copia/incolla
def esegui_verifica_futura():
    global info_ricerca_globale, risultato_text, root, estrazioni_entry_verifica_futura, MAX_COLPI_GIOCO
    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END, "\n\nVerifica esiti futuri (post-analisi) in corso...");
    risultato_text.config(state=tk.DISABLED); root.update_idletasks()

    top_c = info_ricerca_globale.get('top_combinati'); nomi_rv = info_ricerca_globale.get('ruote_verifica'); data_fine = info_ricerca_globale.get('end_date')
    if not all([top_c, nomi_rv, data_fine]):
        messagebox.showerror("Errore Verifica Futura", "Dati analisi 'Successivi' (Top combinati, Ruote verifica, Data Fine) mancanti.");
        risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore Verifica Futura.");risultato_text.config(state=tk.NORMAL); return
    if not any(v for v in top_c.values() if isinstance(v, list) and v):
        messagebox.showinfo("Verifica Futura", "Nessun 'Top Combinato' dall'analisi precedente da verificare.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Nessun Top Combinato.");risultato_text.config(state=tk.NORMAL); return
    try: n_colpi_fut = int(estrazioni_entry_verifica_futura.get()); assert 1 <= n_colpi_fut <= MAX_COLPI_GIOCO
    except:
        messagebox.showerror("Input Invalido", f"N. Colpi Verifica Futura (1-{MAX_COLPI_GIOCO}) non valido.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Input N. Colpi non valido.");risultato_text.config(state=tk.NORMAL); return

    output_verifica = ""
    try:
        res_str = verifica_esiti_futuri(top_c, nomi_rv, data_fine, n_colpi_fut)
        output_verifica = res_str
        if res_str and "Errore" not in res_str and "Nessuna estrazione trovata" not in res_str :
            mostra_popup_testo_semplice("Riepilogo Verifica Predittiva (Post-Analisi)", res_str)
    except Exception as e:
        output_verifica = f"\nErrore durante la verifica esiti futuri: {e}"
        traceback.print_exc()

    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0, tk.END)
    risultato_text.insert(tk.END, output_verifica)
    risultato_text.see(tk.END)

# MODIFICATO per gestione stato risultato_text per copia/incolla
def esegui_verifica_mista():
    global info_ricerca_globale, risultato_text, root, text_combinazioni_miste, estrazioni_entry_verifica_mista, MAX_COLPI_GIOCO
    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
    risultato_text.insert(tk.END, "\n\nVerifica mista (combinazioni utente su trigger spia) in corso...");
    risultato_text.config(state=tk.DISABLED); root.update_idletasks()

    input_text = text_combinazioni_miste.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showerror("Input Invalido", "Nessuna combinazione inserita.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Nessuna combinazione inserita.");risultato_text.config(state=tk.NORMAL); return

    combinazioni_sets = {'estratto': set(),'ambo': set(),'terno': set(),'quaterna': set(),'cinquina': set()}; righe_input_originali = []
    for riga_idx, riga in enumerate(input_text.splitlines()):
        riga_proc = riga.strip();
        if not riga_proc: continue
        righe_input_originali.append(riga_proc)
        try:
            numeri_str = riga_proc.split('-') if '-' in riga_proc else riga_proc.split(); numeri_int = [int(n.strip()) for n in numeri_str if n.strip().isdigit()]
            if not numeri_int: raise ValueError("Nessun numero valido sulla riga.")
            if not all(1<=n<=90 for n in numeri_int): raise ValueError("Numeri fuori range (1-90).")
            if len(set(numeri_int))!=len(numeri_int): raise ValueError("Numeri duplicati.")
            if not (1<=len(numeri_int)<=5): raise ValueError("Inserire da 1 a 5 numeri.")
            numeri_validi_zfill = sorted([str(n).zfill(2) for n in numeri_int]); num_elementi = len(numeri_validi_zfill)
            if num_elementi == 1: combinazioni_sets['estratto'].add(numeri_validi_zfill[0])
            elif num_elementi == 2: combinazioni_sets['ambo'].add(tuple(numeri_validi_zfill))
            elif num_elementi == 3: combinazioni_sets['terno'].add(tuple(numeri_validi_zfill)); _=[combinazioni_sets['ambo'].add(tuple(sorted(ac))) for ac in itertools.combinations(numeri_validi_zfill,2)]
            elif num_elementi == 4: combinazioni_sets['quaterna'].add(tuple(numeri_validi_zfill)); _=[combinazioni_sets['terno'].add(tuple(sorted(tc))) for tc in itertools.combinations(numeri_validi_zfill,3)]; _=[combinazioni_sets['ambo'].add(tuple(sorted(ac))) for ac in itertools.combinations(numeri_validi_zfill,2)]
            elif num_elementi == 5: combinazioni_sets['cinquina'].add(tuple(numeri_validi_zfill)); _=[combinazioni_sets['quaterna'].add(tuple(sorted(qc))) for qc in itertools.combinations(numeri_validi_zfill,4)]; _=[combinazioni_sets['terno'].add(tuple(sorted(tc))) for tc in itertools.combinations(numeri_validi_zfill,3)]; _=[combinazioni_sets['ambo'].add(tuple(sorted(ac))) for ac in itertools.combinations(numeri_validi_zfill,2)]
        except ValueError as ve: messagebox.showerror("Input Invalido",f"Errore riga '{riga_proc}': {ve}");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, f"Errore input riga '{riga_proc}'.");risultato_text.config(state=tk.NORMAL);return
        except Exception as e_parse: messagebox.showerror("Input Invalido",f"Errore riga '{riga_proc}': {e_parse}");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, f"Errore input riga '{riga_proc}'.");risultato_text.config(state=tk.NORMAL);return
    combinazioni_utente = {k: sorted(list(v)) for k, v in combinazioni_sets.items() if v}
    if not any(combinazioni_utente.values()): messagebox.showerror("Input Invalido","Nessuna combinazione valida estratta.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Nessuna combinazione valida.");risultato_text.config(state=tk.NORMAL);return

    date_triggers = info_ricerca_globale.get('date_trigger_ordinate'); nomi_rv = info_ricerca_globale.get('ruote_verifica')
    start_ts = info_ricerca_globale.get('start_date'); end_ts = info_ricerca_globale.get('end_date')
    numeri_spia_originali = info_ricerca_globale.get('numeri_spia_input', [])
    spia_display_originale = format_ambo_terno(numeri_spia_originali) if isinstance(numeri_spia_originali, (list,tuple)) else str(numeri_spia_originali)
    if not all([date_triggers, nomi_rv, start_ts is not None, end_ts is not None]):
        messagebox.showerror("Errore Verifica Mista", "Dati analisi 'Successivi' mancanti.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore Verifica Mista (dati mancanti).");risultato_text.config(state=tk.NORMAL); return
    try: n_colpi_misti = int(estrazioni_entry_verifica_mista.get()); assert 1 <= n_colpi_misti <= MAX_COLPI_GIOCO
    except:
        messagebox.showerror("Input Invalido", f"N. Colpi Verifica Mista (1-{MAX_COLPI_GIOCO}) non valido.");
        risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Input N. Colpi non valido.");risultato_text.config(state=tk.NORMAL); return

    output_verifica_mista = ""
    try:
        titolo_output = f"VERIFICA MISTA (COMBINAZIONI UTENTE) - Dopo Spia: {spia_display_originale}"
        res_str = verifica_esiti_utente_su_triggers(date_triggers, combinazioni_utente, nomi_rv, n_colpi_misti, start_ts, end_ts, titolo_sezione=titolo_output)
        summary_input = "\nInput utente originale (righe elaborate):\n" + "\n".join([f"  - {r}" for r in righe_input_originali])
        lines = res_str.splitlines(); insert_idx_summary = 1
        final_output_lines = lines[:insert_idx_summary] + [summary_input] + lines[insert_idx_summary:]
        output_verifica_mista = "\n".join(final_output_lines)
        if output_verifica_mista and "Errore" not in output_verifica_mista and "Nessun caso trigger" not in output_verifica_mista:
            mostra_popup_testo_semplice(f"Riepilogo Verifica Mista (Spia: {spia_display_originale})", output_verifica_mista)
    except Exception as e:
        output_verifica_mista = f"\nErrore durante la verifica mista: {e}"
        traceback.print_exc()

    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END, output_verifica_mista)
    risultato_text.see(tk.END)


# =============================================================================
# Funzione Wrapper per Visualizza Grafici (INVARIATA)
# =============================================================================
def visualizza_grafici_successivi():
    global risultati_globali, info_ricerca_globale
    if info_ricerca_globale and 'ruote_verifica' in info_ricerca_globale and bool(risultati_globali) and any(r[2] for r in risultati_globali if len(r)>2):
        valid_res = [r for r in risultati_globali if r[2] is not None]
        if valid_res: visualizza_grafici(valid_res, info_ricerca_globale, info_ricerca_globale.get('n_estrazioni',5))
        else: messagebox.showinfo("Grafici", "Nessun risultato valido per grafici.")
    else: messagebox.showinfo("Grafici", "Esegui 'Cerca Successivi' con risultati validi prima.")


# =============================================================================
# FUNZIONI PER NUMERI SIMPATICI (Logica INVARIATA, gestione risultato_text MODIFICATA)
# =============================================================================
def trova_abbinamenti_numero_target(numero_target_str, nomi_ruote_ricerca, start_ts, end_ts, top_n_simpatici):
    global file_ruote
    colonne_numeri = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']
    if not numero_target_str or not (numero_target_str.isdigit() and 1 <= int(numero_target_str) <= 90): return None, "Numero target non valido (deve essere 1-90).", 0
    numero_target_zfill = numero_target_str.zfill(2); abbinamenti_counter = Counter(); occorrenze_target = 0; ruote_effettivamente_analizzate = []
    for nome_ruota in nomi_ruote_ricerca:
        if nome_ruota not in file_ruote: print(f"Attenzione: File per ruota {nome_ruota} non trovato. Saltata."); continue
        df_ruota = carica_dati(file_ruote.get(nome_ruota), start_date=start_ts, end_date=end_ts)
        if df_ruota is None or df_ruota.empty: continue
        ruote_effettivamente_analizzate.append(nome_ruota)
        for _, row in df_ruota.iterrows():
            numeri_estratti_riga = sorted([row[col] for col in colonne_numeri if pd.notna(row[col])])
            if numero_target_zfill in numeri_estratti_riga:
                occorrenze_target += 1
                altri_numeri_nella_stessa_estrazione = [n for n in numeri_estratti_riga if n != numero_target_zfill]
                if altri_numeri_nella_stessa_estrazione: abbinamenti_counter.update(altri_numeri_nella_stessa_estrazione)
    if not ruote_effettivamente_analizzate: return None, "Nessuna ruota selezionata conteneva dati validi.", 0
    if occorrenze_target == 0: return [], f"Numero target '{numero_target_zfill}' non trovato.", 0
    return abbinamenti_counter.most_common(top_n_simpatici), None, occorrenze_target

# MODIFICATO per gestione stato risultato_text per copia/incolla
def esegui_ricerca_numeri_simpatici():
    global risultato_text, root, entry_numero_target_simpatici, listbox_ruote_simpatici
    global entry_top_n_simpatici, start_date_entry, end_date_entry

    if not mappa_file_ruote() or not file_ruote:
        messagebox.showerror("Errore Cartella", "Impossibile leggere i file dalla cartella.\nAssicurati di aver selezionato una cartella con 'Sfoglia...'.")
        risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END); risultato_text.insert(tk.END, "Errore: Cartella o file non validi."); risultato_text.config(state=tk.NORMAL); return

    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
    risultato_text.insert(tk.END, "Ricerca Numeri Simpatici in corso...\n");
    risultato_text.config(state=tk.DISABLED); root.update_idletasks()

    try:
        numero_target = entry_numero_target_simpatici.get().strip()
        if not numero_target or not numero_target.isdigit() or not (1 <= int(numero_target) <= 90): messagebox.showerror("Input Invalido", "Numero Target deve essere tra 1 e 90.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Numero Target non valido.");risultato_text.config(state=tk.NORMAL);return
        selected_ruote_indices = listbox_ruote_simpatici.curselection(); nomi_ruote_selezionate_final = []
        all_ruote_in_listbox = [listbox_ruote_simpatici.get(i) for i in range(listbox_ruote_simpatici.size())]; valid_ruote_from_listbox = [r for r in all_ruote_in_listbox if r not in ["Nessun file valido", "Nessun file ruota valido"]]
        if not selected_ruote_indices:
            if not valid_ruote_from_listbox: messagebox.showerror("Input Invalido", "Nessuna ruota valida disponibile.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Ruote non valide.");risultato_text.config(state=tk.NORMAL);return
            nomi_ruote_selezionate_final = valid_ruote_from_listbox
        else: nomi_ruote_selezionate_final = [listbox_ruote_simpatici.get(i) for i in selected_ruote_indices]
        top_n_str = entry_top_n_simpatici.get().strip()
        if not top_n_str.isdigit() or int(top_n_str) <= 0: messagebox.showerror("Input Invalido", "Numero di 'Top N Simpatici' deve essere un intero positivo.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Top N non valido.");risultato_text.config(state=tk.NORMAL);return
        top_n = int(top_n_str)
        start_dt = start_date_entry.get_date(); end_dt = end_date_entry.get_date()
        if start_dt > end_dt: messagebox.showerror("Input Date", "Data di inizio non può essere successiva alla data di fine.");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, "Errore: Date non valide.");risultato_text.config(state=tk.NORMAL);return
        start_ts = pd.Timestamp(start_dt); end_ts = pd.Timestamp(end_dt)
    except Exception as e: messagebox.showerror("Errore Input", f"Errore input: {e}");risultato_text.config(state=tk.NORMAL);risultato_text.delete(1.0,tk.END);risultato_text.insert(tk.END, f"Errore input: {e}");risultato_text.config(state=tk.NORMAL);return

    risultati_simpatici, errore_msg, occorrenze_target = trova_abbinamenti_numero_target(numero_target, nomi_ruote_selezionate_final, start_ts, end_ts, top_n)

    output_lines = [f"=== Risultati Ricerca Numeri Simpatici ===", f"Numero Target Analizzato: {numero_target.zfill(2)}", f"Ruote Analizzate: {', '.join(nomi_ruote_selezionate_final) if nomi_ruote_selezionate_final else 'Nessuna ruota specificata o valida'}", f"Periodo: {start_dt.strftime('%d/%m/%Y')} - {end_dt.strftime('%d/%m/%Y')}"]
    if errore_msg: output_lines.append(f"\nMessaggio dal sistema: {errore_msg}")
    if risultati_simpatici is None: output_lines.append("Ricerca fallita o interrotta.")
    elif occorrenze_target == 0:
        if not errore_msg: output_lines.append(f"Il Numero Target '{numero_target.zfill(2)}' non è stato trovato.")
    else:
        output_lines.append(f"Il Numero Target '{numero_target.zfill(2)}' è stato trovato {occorrenze_target} volte.")
        if not risultati_simpatici: output_lines.append("Nessun altro numero abbinato trovato.")
        else:
            output_lines.append(f"\nTop {len(risultati_simpatici)} Numeri Simpatici (Numero: Frequenza Abbinamento):")
            for i, (num_simp, freq) in enumerate(risultati_simpatici): output_lines.append(f"  {i+1}. Numero {str(num_simp).zfill(2)}: {freq} volte")

    final_output_str = "\n".join(output_lines)
    risultato_text.config(state=tk.NORMAL)
    risultato_text.delete(1.0,tk.END)
    risultato_text.insert(tk.END, final_output_str)
    risultato_text.see(tk.END)

    if not errore_msg and risultati_simpatici is not None and occorrenze_target > 0:
         mostra_popup_testo_semplice(f"Numeri Simpatici per {numero_target.zfill(2)}", final_output_str)
    elif errore_msg and (risultati_simpatici is not None or occorrenze_target == 0):
        mostra_popup_testo_semplice(f"Info Numeri Simpatici per {numero_target.zfill(2)}", final_output_str)


# =============================================================================
# GUI e Mainloop (INVARIATE rispetto alla versione precedente,
# tranne il messaggio di benvenuto per riflettere il numero di colpi)
# =============================================================================
root = tk.Tk()
root.title("Numeri Spia - Ambo spia - Spia in posizione- Marker e Simpatici- Created by Il Lotto di Max")
root.geometry("1350x850")
root.minsize(1200, 750)
root.configure(bg="#f0f0f0")

style = ttk.Style(); style.theme_use('clam')
style.configure("TFrame", background="#f0f0f0"); style.configure("TLabel", background="#f0f0f0", font=("Segoe UI",10))
style.configure("TButton", font=("Segoe UI",10), padding=5); style.configure("Title.TLabel", font=("Segoe UI",11,"bold"))
style.configure("Header.TLabel", font=("Segoe UI",12,"bold")); style.configure("Small.TLabel", background="#f0f0f0", font=("Segoe UI",8))
style.configure("TEntry", padding=3); style.configure("TListbox", font=("Consolas",10)); style.configure("TLabelframe.Label", font=("Segoe UI",10,"bold"), background="#f0f0f0")
style.configure("TNotebook.Tab", padding=[10,5], font=("Segoe UI",10))
style.configure("TRadiobutton", background="#f0f0f0", font=("Segoe UI", 9))

main_frame = ttk.Frame(root, padding=10); main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
cartella_frame = ttk.Frame(main_frame); cartella_frame.pack(fill=tk.X, pady=(0,10))
ttk.Label(cartella_frame, text="Cartella Estrazioni:", style="Title.TLabel").pack(side=tk.LEFT, padx=(0,5))
cartella_entry = ttk.Entry(cartella_frame, width=60); cartella_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
btn_sfoglia = ttk.Button(cartella_frame, text="Sfoglia...")
btn_sfoglia.pack(side=tk.LEFT, padx=5)

notebook = ttk.Notebook(main_frame, style="TNotebook"); notebook.pack(fill=tk.X, pady=10)

tab_successivi = ttk.Frame(notebook, padding=10)
notebook.add(tab_successivi, text=' Analisi Numeri Successivi (Spia) ')
controls_frame_succ = ttk.Frame(tab_successivi)
controls_frame_succ.pack(fill=tk.X)
controls_frame_succ.columnconfigure(0, weight=1); controls_frame_succ.columnconfigure(1, weight=1); controls_frame_succ.columnconfigure(2, weight=0); controls_frame_succ.columnconfigure(3, weight=0)
ruote_analisi_outer_frame = ttk.Frame(controls_frame_succ); ruote_analisi_outer_frame.grid(row=0, column=0, sticky="nsew", padx=(0,5))
ttk.Label(ruote_analisi_outer_frame, text="1. Ruote Analisi (Spia):", style="Title.TLabel").pack(anchor="w"); ttk.Label(ruote_analisi_outer_frame, text="(CTRL/SHIFT)", style="Small.TLabel").pack(anchor="w",pady=(0,5))
ruote_analisi_list_frame = ttk.Frame(ruote_analisi_outer_frame); ruote_analisi_list_frame.pack(fill=tk.BOTH, expand=True)
scrollbar_ra = ttk.Scrollbar(ruote_analisi_list_frame); scrollbar_ra.pack(side=tk.RIGHT, fill=tk.Y)
listbox_ruote_analisi = tk.Listbox(ruote_analisi_list_frame, height=10, selectmode=tk.EXTENDED, exportselection=False, font=("Consolas",10), selectbackground="#005A9E", selectforeground="white", yscrollcommand=scrollbar_ra.set)
listbox_ruote_analisi.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar_ra.config(command=listbox_ruote_analisi.yview)
ruote_verifica_outer_frame = ttk.Frame(controls_frame_succ); ruote_verifica_outer_frame.grid(row=0, column=1, sticky="nsew", padx=5)
ttk.Label(ruote_verifica_outer_frame, text="4. Ruote Verifica (Esiti):", style="Title.TLabel").pack(anchor="w"); ttk.Label(ruote_verifica_outer_frame, text="(CTRL/SHIFT)", style="Small.TLabel").pack(anchor="w",pady=(0,5))
ruote_verifica_list_frame = ttk.Frame(ruote_verifica_outer_frame); ruote_verifica_list_frame.pack(fill=tk.BOTH, expand=True)
scrollbar_rv = ttk.Scrollbar(ruote_verifica_list_frame); scrollbar_rv.pack(side=tk.RIGHT, fill=tk.Y)
listbox_ruote_verifica = tk.Listbox(ruote_verifica_list_frame, height=10, selectmode=tk.EXTENDED, exportselection=False, font=("Consolas",10), selectbackground="#005A9E", selectforeground="white", yscrollcommand=scrollbar_rv.set)
listbox_ruote_verifica.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar_rv.config(command=listbox_ruote_verifica.yview)
center_controls_frame_succ = ttk.Frame(controls_frame_succ); center_controls_frame_succ.grid(row=0, column=2, sticky="ns", padx=5)
tipo_spia_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 2. Tipo di Spia ", padding=5); tipo_spia_frame_succ.pack(fill=tk.X, pady=(0,5))
if 'tipo_spia_var_global' not in globals() or globals()['tipo_spia_var_global'] is None: tipo_spia_var_global = tk.StringVar(value="estratto")
def _toggle_posizione_spia_input_tab_succ():
    global tipo_spia_var_global, combo_posizione_spia
    is_posizionale = (tipo_spia_var_global.get() == "estratto_posizionale")
    if combo_posizione_spia: combo_posizione_spia.config(state="readonly" if is_posizionale else tk.DISABLED);_=[combo_posizione_spia.current(0) if is_posizionale and (not combo_posizione_spia.get() or combo_posizione_spia.get() not in combo_posizione_spia['values']) else None, combo_posizione_spia.set("Qualsiasi") if not is_posizionale else None]
ttk.Radiobutton(tipo_spia_frame_succ, text="Estratto Spia", variable=tipo_spia_var_global, value="estratto", style="TRadiobutton", command=_toggle_posizione_spia_input_tab_succ).pack(anchor="w", padx=5)
ttk.Radiobutton(tipo_spia_frame_succ, text="Ambo Spia", variable=tipo_spia_var_global, value="ambo", style="TRadiobutton", command=_toggle_posizione_spia_input_tab_succ).pack(anchor="w", padx=5)
ttk.Radiobutton(tipo_spia_frame_succ, text="Estratto Spia Posizionale", variable=tipo_spia_var_global, value="estratto_posizionale", style="TRadiobutton", command=_toggle_posizione_spia_input_tab_succ).pack(anchor="w", padx=5)
spia_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 3. Numeri Spia / Posizione ",padding=5); spia_frame_succ.pack(fill=tk.X,pady=(0,5))
spia_entry_container_succ = ttk.Frame(spia_frame_succ); spia_entry_container_succ.pack(fill=tk.X,pady=5)
entry_numeri_spia.clear(); _=[entry_numeri_spia.append(ttk.Entry(spia_entry_container_succ,width=5,justify=tk.CENTER,font=("Segoe UI",10))) or entry_numeri_spia[-1].pack(side=tk.LEFT,padx=3,ipady=2) for _ in range(5)]
posizioni_spia_options = ["Qualsiasi", "1a", "2a", "3a", "4a", "5a"]
if 'combo_posizione_spia' not in globals() or globals()['combo_posizione_spia'] is None: combo_posizione_spia = ttk.Combobox(spia_entry_container_succ, values=posizioni_spia_options, width=8, font=("Segoe UI", 9), state="disabled"); combo_posizione_spia.set("Qualsiasi"); combo_posizione_spia.pack(side=tk.LEFT, padx=(5,0), ipady=1)
estrazioni_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 5. Estrazioni Successive ",padding=5); estrazioni_frame_succ.pack(fill=tk.X,pady=5)
ttk.Label(estrazioni_frame_succ, text=f"Quante (1-{MAX_COLPI_GIOCO}):", style="Small.TLabel").pack(anchor="w")
estrazioni_entry_succ = ttk.Entry(estrazioni_frame_succ,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_succ.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_succ.insert(0,"5"); _toggle_posizione_spia_input_tab_succ()
buttons_frame_succ = ttk.Frame(controls_frame_succ); buttons_frame_succ.grid(row=0, column=3, sticky="ns", padx=(10,0))
button_cerca_succ = ttk.Button(buttons_frame_succ, text="Cerca Successivi", command=lambda:cerca_numeri(modalita="successivi")); button_cerca_succ.pack(pady=5, fill=tk.X, ipady=3)
tab_antecedenti = ttk.Frame(notebook,padding=10); notebook.add(tab_antecedenti, text=' Analisi Numeri Antecedenti (Marker) ')
controls_frame_ant = ttk.Frame(tab_antecedenti); controls_frame_ant.pack(fill=tk.X); controls_frame_ant.columnconfigure(0,weight=1); controls_frame_ant.columnconfigure(1,weight=0); controls_frame_ant.columnconfigure(2,weight=0)
ruote_analisi_ant_outer_frame = ttk.Frame(controls_frame_ant); ruote_analisi_ant_outer_frame.grid(row=0,column=0,sticky="nsew",padx=(0,10)); ttk.Label(ruote_analisi_ant_outer_frame, text="1. Ruote da Analizzare:",style="Title.TLabel").pack(anchor="w"); ttk.Label(ruote_analisi_ant_outer_frame, text="(Obiettivo e antecedenti cercati qui)",style="Small.TLabel").pack(anchor="w",pady=(0,5))
ruote_analisi_ant_list_frame = ttk.Frame(ruote_analisi_ant_outer_frame); ruote_analisi_ant_list_frame.pack(fill=tk.BOTH,expand=True); scrollbar_ra_ant = ttk.Scrollbar(ruote_analisi_ant_list_frame); scrollbar_ra_ant.pack(side=tk.RIGHT,fill=tk.Y)
listbox_ruote_analisi_ant = tk.Listbox(ruote_analisi_ant_list_frame,height=10,selectmode=tk.EXTENDED,exportselection=False,font=("Consolas",10),selectbackground="#005A9E",selectforeground="white",yscrollcommand=scrollbar_ra_ant.set); listbox_ruote_analisi_ant.pack(side=tk.LEFT,fill=tk.BOTH,expand=True); scrollbar_ra_ant.config(command=listbox_ruote_analisi_ant.yview)
center_controls_frame_ant = ttk.Frame(controls_frame_ant); center_controls_frame_ant.grid(row=0,column=1,sticky="ns",padx=10); obiettivo_frame_ant = ttk.LabelFrame(center_controls_frame_ant, text=" 2. Numeri Obiettivo (1-90) ",padding=5); obiettivo_frame_ant.pack(fill=tk.X,pady=(0,5))
obiettivo_entry_container_ant = ttk.Frame(obiettivo_frame_ant); obiettivo_entry_container_ant.pack(fill=tk.X,pady=5)
if 'entry_numeri_obiettivo' not in globals() or not isinstance(globals()['entry_numeri_obiettivo'], list): entry_numeri_obiettivo = []
else: entry_numeri_obiettivo.clear()
_=[entry_numeri_obiettivo.append(ttk.Entry(obiettivo_entry_container_ant,width=5,justify=tk.CENTER,font=("Segoe UI",10))) or entry_numeri_obiettivo[-1].pack(side=tk.LEFT,padx=3,ipady=2) for _ in range(5)]
estrazioni_frame_ant = ttk.LabelFrame(center_controls_frame_ant, text=" 3. Estrazioni Precedenti ",padding=5); estrazioni_frame_ant.pack(fill=tk.X,pady=5); ttk.Label(estrazioni_frame_ant, text="Quante controllare (>=1):",style="Small.TLabel").pack(anchor="w")
estrazioni_entry_ant = ttk.Entry(estrazioni_frame_ant,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_ant.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_ant.insert(0,"3")
buttons_frame_ant = ttk.Frame(controls_frame_ant); buttons_frame_ant.grid(row=0,column=2,sticky="ns",padx=(10,0)); button_cerca_ant = ttk.Button(buttons_frame_ant, text="Cerca Antecedenti",command=lambda:cerca_numeri(modalita="antecedenti")); button_cerca_ant.pack(pady=5,fill=tk.X,ipady=3)
tab_simpatici = ttk.Frame(notebook, padding=10); notebook.add(tab_simpatici, text=' Numeri Simpatici ')
controls_frame_simpatici_outer = ttk.Frame(tab_simpatici); controls_frame_simpatici_outer.pack(fill=tk.X, pady=5); controls_frame_simpatici_outer.columnconfigure(0, weight=0); controls_frame_simpatici_outer.columnconfigure(1, weight=1)
input_params_simpatici_frame = ttk.Frame(controls_frame_simpatici_outer); input_params_simpatici_frame.grid(row=0, column=0, sticky="ns", padx=(0,10)); lbl_numero_target = ttk.Label(input_params_simpatici_frame, text="Numero Target (1-90):", style="Title.TLabel"); lbl_numero_target.pack(anchor="w", pady=(0,2))
entry_numero_target_simpatici = ttk.Entry(input_params_simpatici_frame, width=10, justify=tk.CENTER, font=("Segoe UI",10)); entry_numero_target_simpatici.pack(anchor="w", pady=(0,10), ipady=2); entry_numero_target_simpatici.insert(0, "10"); lbl_top_n_simpatici = ttk.Label(input_params_simpatici_frame, text="Quanti Numeri Simpatici (Top N):", style="Title.TLabel"); lbl_top_n_simpatici.pack(anchor="w", pady=(5,2))
entry_top_n_simpatici = ttk.Entry(input_params_simpatici_frame, width=10, justify=tk.CENTER, font=("Segoe UI",10)); entry_top_n_simpatici.pack(anchor="w", pady=(0,10), ipady=2); entry_top_n_simpatici.insert(0, "10"); button_cerca_simpatici = ttk.Button(input_params_simpatici_frame, text="Cerca Numeri Simpatici", command=esegui_ricerca_numeri_simpatici); button_cerca_simpatici.pack(pady=10, fill=tk.X, ipady=3)
ruote_simpatici_frame_outer = ttk.Frame(controls_frame_simpatici_outer); ruote_simpatici_frame_outer.grid(row=0, column=1, sticky="nsew", padx=(10,0)); ttk.Label(ruote_simpatici_frame_outer, text="Ruote di Ricerca (CTRL/SHIFT per multiple, nessuna = tutte):", style="Title.TLabel").pack(anchor="w")
ruote_simpatici_list_frame_inner = ttk.Frame(ruote_simpatici_frame_outer); ruote_simpatici_list_frame_inner.pack(fill=tk.BOTH, expand=True, pady=(5,0)); scrollbar_rs_y = ttk.Scrollbar(ruote_simpatici_list_frame_inner, orient=tk.VERTICAL); scrollbar_rs_y.pack(side=tk.RIGHT, fill=tk.Y)
listbox_ruote_simpatici = tk.Listbox(ruote_simpatici_list_frame_inner, height=10, selectmode=tk.EXTENDED, exportselection=False, font=("Consolas",10), selectbackground="#005A9E", selectforeground="white", yscrollcommand=scrollbar_rs_y.set); listbox_ruote_simpatici.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar_rs_y.config(command=listbox_ruote_simpatici.yview)
common_controls_top_frame = ttk.Frame(main_frame); common_controls_top_frame.pack(fill=tk.X,pady=5)
dates_frame = ttk.LabelFrame(common_controls_top_frame, text=" Periodo Analisi (Comune) ",padding=5); dates_frame.pack(side=tk.LEFT,padx=(0,10),fill=tk.Y); dates_frame.columnconfigure(1,weight=1)
ttk.Label(dates_frame,text="Da:",anchor="e").grid(row=0,column=0,padx=2,pady=2,sticky="w"); start_date_default = datetime.date.today()-datetime.timedelta(days=365*3)
start_date_entry = DateEntry(dates_frame,width=10,background='#3498db',foreground='white',borderwidth=2,date_pattern='yyyy-mm-dd',font=("Segoe UI",9),year=start_date_default.year,month=start_date_default.month,day=start_date_default.day); start_date_entry.grid(row=0,column=1,padx=2,pady=2,sticky="ew")
ttk.Label(dates_frame,text="A:",anchor="e").grid(row=1,column=0,padx=2,pady=2,sticky="w"); end_date_entry = DateEntry(dates_frame,width=10,background='#3498db',foreground='white',borderwidth=2,date_pattern='yyyy-mm-dd',font=("Segoe UI",9)); end_date_entry.grid(row=1,column=1,padx=2,pady=2,sticky="ew")
common_buttons_frame = ttk.Frame(common_controls_top_frame); common_buttons_frame.pack(side=tk.LEFT,padx=10,fill=tk.Y)
button_salva = ttk.Button(common_buttons_frame,text="Salva Risultati",command=salva_risultati); button_salva.pack(side=tk.LEFT,pady=5,padx=5,ipady=3)
button_visualizza = ttk.Button(common_buttons_frame,text="Visualizza Grafici\n(Solo Successivi)",command=visualizza_grafici_successivi); button_visualizza.pack(side=tk.LEFT,pady=5,padx=5,ipady=0); button_visualizza.config(state=tk.DISABLED)
post_analysis_checks_frame = ttk.Frame(main_frame); post_analysis_checks_frame.pack(fill=tk.X,pady=(5,0))
verifica_futura_frame = ttk.LabelFrame(post_analysis_checks_frame, text=" Verifica Predittiva (Post-Analisi) ",padding=5); verifica_futura_frame.pack(side=tk.LEFT,padx=(0,10),fill=tk.Y,expand=False)
ttk.Label(verifica_futura_frame,text=f"Controlla N Colpi (1-{MAX_COLPI_GIOCO}):",style="Small.TLabel").pack(anchor="w")
estrazioni_entry_verifica_futura = ttk.Entry(verifica_futura_frame,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_verifica_futura.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_verifica_futura.insert(0,"9")
button_verifica_futura = ttk.Button(verifica_futura_frame,text="Verifica Futura\n(Post-Analisi)",command=esegui_verifica_futura); button_verifica_futura.pack(pady=5,fill=tk.X,ipady=0); button_verifica_futura.config(state=tk.DISABLED)
verifica_mista_frame = ttk.LabelFrame(post_analysis_checks_frame, text=" Verifica Mista (su Trigger Spia) ",padding=5); verifica_mista_frame.pack(side=tk.LEFT,padx=10,fill=tk.BOTH,expand=True)
ttk.Label(verifica_mista_frame, text="Combinazioni (1-5 numeri, 1-90, una per riga, separate da '-' o spazio):", style="Small.TLabel").pack(anchor="w")
text_mista_container = ttk.Frame(verifica_mista_frame); text_mista_container.pack(fill=tk.BOTH, expand=True, pady=(0,5)); text_combinazioni_miste_scrollbar_y = ttk.Scrollbar(text_mista_container, orient=tk.VERTICAL); text_combinazioni_miste_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
text_combinazioni_miste = tk.Text(text_mista_container, height=4, width=30, font=("Consolas", 10), wrap=tk.WORD, yscrollcommand=text_combinazioni_miste_scrollbar_y.set, bd=1, relief="sunken"); text_combinazioni_miste.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); text_combinazioni_miste_scrollbar_y.config(command=text_combinazioni_miste.yview); text_combinazioni_miste.insert("1.0", "18 23 43\n60")
ttk.Label(verifica_mista_frame, text=f"Controlla N Colpi Verifica (1-{MAX_COLPI_GIOCO}):", style="Small.TLabel").pack(anchor="w")
estrazioni_entry_verifica_mista = ttk.Entry(verifica_mista_frame,width=5,justify=tk.CENTER,font=("Segoe UI",10)); estrazioni_entry_verifica_mista.pack(anchor="w",pady=2,ipady=2); estrazioni_entry_verifica_mista.insert(0,"9")
button_verifica_mista = ttk.Button(verifica_mista_frame,text="Verifica Mista\n(su Trigger Spia)",command=esegui_verifica_mista); button_verifica_mista.pack(pady=5,fill=tk.X,ipady=0); button_verifica_mista.config(state=tk.DISABLED)
ttk.Label(main_frame, text="Risultati Analisi (Log):", style="Header.TLabel").pack(anchor="w",pady=(15,0))
risultato_outer_frame = ttk.Frame(main_frame); risultato_outer_frame.pack(fill=tk.BOTH,expand=True,pady=5); risultato_scroll_y = ttk.Scrollbar(risultato_outer_frame,orient=tk.VERTICAL); risultato_scroll_y.pack(side=tk.RIGHT,fill=tk.Y); risultato_scroll_x = ttk.Scrollbar(risultato_outer_frame,orient=tk.HORIZONTAL); risultato_scroll_x.pack(side=tk.BOTTOM,fill=tk.X)
risultato_text = tk.Text(risultato_outer_frame,wrap=tk.NONE,font=("Consolas",10),height=15,yscrollcommand=risultato_scroll_y.set,xscrollcommand=risultato_scroll_x.set,state=tk.NORMAL,bd=1,relief="sunken")
risultato_text.pack(fill=tk.BOTH,expand=True); risultato_scroll_y.config(command=risultato_text.yview); risultato_scroll_x.config(command=risultato_text.xview)

def aggiorna_lista_file_gui(target_listbox):
    global file_ruote
    if not target_listbox: return
    target_listbox.config(state=tk.NORMAL); target_listbox.delete(0, tk.END)

    # Modifica per ordinamento personalizzato: NAZIONALE per ultima
    lista_ruote_originale = list(file_ruote.keys())
    ruota_nazionale_str = "NAZIONALE"

    ruote_ordinate = sorted([r for r in lista_ruote_originale if r != ruota_nazionale_str])

    if ruota_nazionale_str in lista_ruote_originale:
        ruote_ordinate.append(ruota_nazionale_str)
    # Fine modifica

    if ruote_ordinate:
        for r in ruote_ordinate:
            target_listbox.insert(tk.END, r)
    else:
        target_listbox.insert(tk.END, "Nessun file ruota valido")
        target_listbox.config(state=tk.DISABLED)

def mappa_file_ruote():
    global file_ruote, cartella_entry
    cartella = cartella_entry.get(); file_ruote.clear()
    if not cartella or not os.path.isdir(cartella): return False
    ruote_valide = ['BARI','CAGLIARI','FIRENZE','GENOVA','MILANO','NAPOLI','PALERMO','ROMA','TORINO','VENEZIA','NAZIONALE']; found = False
    try:
        for file in os.listdir(cartella):
            fp = os.path.join(cartella, file)
            if os.path.isfile(fp) and file.lower().endswith(".txt"):
                nome_base = os.path.splitext(file)[0].upper()
                if nome_base in ruote_valide: file_ruote[nome_base] = fp; found = True
        return found
    except OSError as e: print(f"Errore lettura cartella: {e}"); return False
    except Exception as e: print(f"Errore scansione file: {e}"); traceback.print_exc(); return False

def on_sfoglia_click():
    global cartella_entry, listbox_ruote_analisi, listbox_ruote_verifica, listbox_ruote_analisi_ant, listbox_ruote_simpatici
    cartella_sel = filedialog.askdirectory(title="Seleziona Cartella Estrazioni")
    if cartella_sel:
        cartella_entry.delete(0,tk.END); cartella_entry.insert(0,cartella_sel)
        if mappa_file_ruote(): _=[aggiorna_lista_file_gui(lb) for lb in [listbox_ruote_analisi, listbox_ruote_verifica, listbox_ruote_analisi_ant, listbox_ruote_simpatici] if lb]
        else: _=[(lb.config(state=tk.NORMAL),lb.delete(0,tk.END),lb.insert(tk.END,"Nessun file valido"),lb.config(state=tk.DISABLED)) for lb in [listbox_ruote_analisi, listbox_ruote_verifica, listbox_ruote_analisi_ant, listbox_ruote_simpatici] if lb]; messagebox.showwarning("Nessun File", "Nessun file .txt valido trovato.")
btn_sfoglia.config(command=on_sfoglia_click)

def main():
    global root, risultato_text, MAX_COLPI_GIOCO
    risultato_text.config(state=tk.NORMAL); risultato_text.delete(1.0, tk.END)
    welcome_message = (
        f"Benvenuto in Numeri Spia (Colpi Max: {MAX_COLPI_GIOCO}, Esito 'IN CORSO', Copia/Salva Popup)!\n\n"
        "1. Usa 'Sfoglia...' per selezionare la cartella delle estrazioni.\n"
        "2. Imposta il periodo di analisi.\n"
        "3. Scegli la modalità di analisi:\n"
        "   - 'Analisi Numeri Successivi (Spia)':\n"
        "     - Ruote Analisi (spia) e Verifica (esiti).\n"
        "     - Tipo Spia (Estratto, Ambo, Estratto Posizionale).\n"
        f"     - Estrazioni Successive da controllare (1-{MAX_COLPI_GIOCO}).\n"
        "     - Clicca 'Cerca Successivi'.\n"
        "   - 'Analisi Numeri Antecedenti (Marker)': (Dettagli nella tab)\n"
        "   - 'Numeri Simpatici': (Dettagli nella tab)\n"
        f"\nFunzioni di Verifica (usano i risultati dell'analisi 'Successivi'):\n"
        f" - 'Verifica Predittiva': controlla esiti per N colpi (1-{MAX_COLPI_GIOCO}) dopo la data fine analisi.\n"
        f" - 'Verifica Mista': controlla combinazioni personalizzate per N colpi (1-{MAX_COLPI_GIOCO}) dopo ogni trigger spia.\n"
        "   (NOVITÀ) Esito 'IN CORSO' se i dati finiscono prima dei colpi richiesti.\n"
        "\nI risultati nel box sottostante sono copiabili. I popup hanno un pulsante 'Salva su File...'."
    )
    risultato_text.insert(tk.END, welcome_message)
    root.mainloop()
    print("\nScript terminato.")

if __name__ == "__main__":
    main()