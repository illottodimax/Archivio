# main_app.py
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
from datetime import datetime
from collections import Counter
import os

try:
    from tkcalendar import DateEntry
except ImportError:
    messagebox.showerror("Libreria Mancante", "La libreria 'tkcalendar' non è installata.\n\nPer favore, installala con: pip install tkcalendar")
    exit()

from archivio_lotto import ArchivioLotto
from metodo_mensile import MetodoMensile, Correttore

class MetodoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Metodo Mensile con Ottimizzatore e Validatore")
        self.geometry("800x850")

        self.output_queue = queue.Queue()
        self.archivio = ArchivioLotto(self.output_queue)
        self.metodo = MetodoMensile()
        
        self.date_fine_mese = []
        self.casi_negativi_per_correttore = []
        self.ultimo_caso_analizzato = None
        self.miglior_correttore_trovato = None
        self.backtest_params = {}

        self._create_widgets()
        self.after(100, self._process_queue)

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Sezione 1: Inizializzazione
        init_frame = ttk.LabelFrame(main_frame, text="1. Inizializzazione", padding=10)
        init_frame.pack(fill="x", pady=5)
        source_frame = ttk.Frame(init_frame)
        source_frame.pack(fill="x", pady=2)
        ttk.Label(source_frame, text="Fonte Dati:").pack(side="left", padx=5)
        self.data_source_var = tk.StringVar(value='GitHub')
        ttk.Radiobutton(source_frame, text="GitHub", variable=self.data_source_var, value='GitHub', command=self._toggle_local_path).pack(side="left")
        ttk.Radiobutton(source_frame, text="Locale", variable=self.data_source_var, value='Locale', command=self._toggle_local_path).pack(side="left", padx=5)
        self.local_path_label = ttk.Label(source_frame, text="N/A")
        self.local_path_label.pack(side="left", padx=5)
        self.browse_button = ttk.Button(source_frame, text="Sfoglia...", command=self._select_local_path)
        self.browse_button.pack(side="left")
        self.init_button = ttk.Button(init_frame, text="Inizializza Archivio", command=self._run_initialize)
        self.init_button.pack(fill="x", pady=(10, 2))

        # Sezione 2: Calcolo su Data Specifica
        single_calc_frame = ttk.LabelFrame(main_frame, text="2. Calcolo su Data Specifica", padding=10)
        single_calc_frame.pack(fill="x", pady=5)
        ttk.Label(single_calc_frame, text="Seleziona una data di calcolo:").pack(side="left", padx=5)
        self.single_date_entry = DateEntry(single_calc_frame, width=12, date_pattern='yyyy-mm-dd', locale='it_IT')
        self.single_date_entry.pack(side="left", padx=5)
        self.single_date_entry.config(state="disabled")
        self.calc_button = ttk.Button(single_calc_frame, text="Calcola Previsione", state="disabled", command=self._run_single_prediction)
        self.calc_button.pack(side="left", padx=10)

        # Sezione 3: Analisi Storica
        backtest_frame = ttk.LabelFrame(main_frame, text="3. Analisi Storica (solo Fine Mese)", padding=10)
        backtest_frame.pack(fill="x", pady=5)
        ttk.Label(backtest_frame, text="Periodo Dal:").grid(row=0, column=0, padx=5, pady=5)
        self.backtest_start_date = DateEntry(backtest_frame, width=12, date_pattern='yyyy-mm-dd', locale='it_IT')
        self.backtest_start_date.grid(row=0, column=1, padx=5, pady=5)
        self.backtest_start_date.config(state="disabled")
        ttk.Label(backtest_frame, text="Al:").grid(row=0, column=2, padx=5, pady=5)
        self.backtest_end_date = DateEntry(backtest_frame, width=12, date_pattern='yyyy-mm-dd', locale='it_IT')
        self.backtest_end_date.grid(row=0, column=3, padx=5, pady=5)
        self.backtest_end_date.config(state="disabled")
        ttk.Label(backtest_frame, text="Colpi di Gioco:").grid(row=1, column=0, padx=5, pady=5)
        self.colpi_spinbox = ttk.Spinbox(backtest_frame, from_=1, to=25, width=5)
        self.colpi_spinbox.set(13)
        self.colpi_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.colpi_spinbox.config(state="disabled")
        self.backtest_button = ttk.Button(backtest_frame, text="Esegui Analisi Storica", state="disabled", command=self._run_backtest)
        self.backtest_button.grid(row=1, column=2, columnspan=3, padx=10, pady=5, sticky="ew")
        
        # Sezione 4: Ottimizzazione
        optimizer_frame = ttk.LabelFrame(main_frame, text="4. Ottimizzazione (Correttore)", padding=10)
        optimizer_frame.pack(fill="x", pady=5)
        self.corrector_button = ttk.Button(optimizer_frame, text="Trova Correttore Ottimale per Ambata", state="disabled", command=self._run_corrector)
        self.corrector_button.pack(fill="x", pady=2)
        
        # NUOVA SEZIONE 5: VALIDAZIONE
        validator_frame = ttk.LabelFrame(main_frame, text="5. Validazione", padding=10)
        validator_frame.pack(fill="x", pady=5)
        self.validator_button = ttk.Button(validator_frame, text="Valida Miglior Correttore (su tutto il periodo)", state="disabled", command=self._run_validator)
        self.validator_button.pack(fill="x", pady=2)
        
        # Pannello Output
        output_frame = ttk.LabelFrame(main_frame, text="Risultati", padding=10)
        output_frame.pack(fill="both", expand=True, pady=5)
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state='disabled', font=('Consolas', 10))
        self.output_text.pack(fill="both", expand=True)
        ttk.Button(output_frame, text="Pulisci Output", command=self._clear_output).pack(anchor='se', pady=(5,0))

        self._toggle_local_path()

    def _log(self, message): self.output_queue.put(message)
    def _clear_output(self):
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')
    def _run_in_thread(self, target_func): threading.Thread(target=target_func, daemon=True).start()
    def _select_local_path(self):
        folder = filedialog.askdirectory()
        if folder: self.archivio.local_path, self.local_path_label.config(text=os.path.basename(folder))
    def _toggle_local_path(self):
        state = 'normal' if self.data_source_var.get() == 'Locale' else 'disabled'
        self.local_path_label.config(state=state)
        self.browse_button.config(state=state)
        self.archivio.data_source = self.data_source_var.get()

    def _run_initialize(self): self._clear_output(); self._run_in_thread(self._initialize_task)
    def _initialize_task(self):
        try:
            self.archivio.inizializza()
            self._trova_date_fine_mese()
            self._log("\nDate di fine mese calcolate.")
            if self.archivio.date_ordinate:
                first_date = datetime.strptime(self.archivio.date_ordinate[0], '%Y-%m-%d')
                last_date = datetime.strptime(self.archivio.date_ordinate[-1], '%Y-%m-%d')
                self.single_date_entry.set_date(last_date)
                self.backtest_start_date.set_date(first_date)
                self.backtest_end_date.set_date(last_date)
                for w in [self.single_date_entry, self.backtest_start_date, self.backtest_end_date, self.calc_button, self.backtest_button, self.colpi_spinbox]: w.config(state="normal")
                for w in [self.corrector_button, self.validator_button]: w.config(state="disabled")
        except Exception as e: self._log(f"\nERRORE durante l'inizializzazione: {e}")

    def _trova_date_fine_mese(self):
        date_ordinate = self.archivio.date_ordinate
        self.date_fine_mese = []
        if not date_ordinate: return
        for i in range(len(date_ordinate) - 1):
            if date_ordinate[i][5:7] != date_ordinate[i+1][5:7]: self.date_fine_mese.append(date_ordinate[i])

    def _run_single_prediction(self): self._clear_output(); self._run_in_thread(self._single_prediction_task)
    def _single_prediction_task(self):
        try:
            data_selezionata = self.single_date_entry.get_date().strftime('%Y-%m-%d')
            if data_selezionata not in self.archivio.dati_per_analisi:
                self._log(f"Errore: la data {data_selezionata} non è una data di estrazione valida."); return
        except Exception as e: self._log(f"Errore nella data selezionata: {e}"); return
        self._log(f"--- CALCOLO PREVISIONE PER LA DATA SELEZIONATA ---\n\nData di calcolo: {data_selezionata}\n")
        previsione = self.metodo.calcola_previsione(self.archivio.dati_per_analisi[data_selezionata])
        if previsione: self.ultimo_caso_analizzato = previsione; self._stampa_previsione(previsione)
        else: self._log("Impossibile calcolare la previsione: dati mancanti per NA o PA in quella data.")

    def _run_backtest(self): self._clear_output(); self._run_in_thread(self._backtest_task_wrapper)
    def _backtest_task_wrapper(self):
        self.casi_negativi_per_correttore, self.ultimo_caso_analizzato, self.miglior_correttore_trovato = [], None, None
        for w in [self.corrector_button, self.validator_button]: w.config(state="disabled")
        try:
            self.backtest_params = {
                'start_date_str': self.backtest_start_date.get_date().strftime('%Y-%m-%d'),
                'end_date_str': self.backtest_end_date.get_date().strftime('%Y-%m-%d'),
                'colpi_da_giocare': int(self.colpi_spinbox.get())
            }
        except Exception as e: self._log(f"Errore nei parametri del backtest: {e}"); return
        self._log(f"--- AVVIO ANALISI STORICA (dal {self.backtest_params['start_date_str']} al {self.backtest_params['end_date_str']}) ---\n")
        self._esegui_backtest_concreto(fisso=72)
        if self.casi_negativi_per_correttore:
            self.corrector_button.config(state="normal")
            self._log(f"\nATTENZIONE: Trovati {len(self.casi_negativi_per_correttore)} casi negativi. Ora puoi usare il 'Correttore Ottimale'.")

    def _esegui_backtest_concreto(self, fisso=72, silent=False):
        date_da_analizzare = [d for d in self.date_fine_mese if self.backtest_params['start_date_str'] <= d <= self.backtest_params['end_date_str']]
        if not date_da_analizzare:
            if not silent: self._log("Nessuna estrazione di fine mese trovata nel periodo selezionato."); return None
        
        vincite = Counter()
        for data_calcolo in date_da_analizzare:
            previsione = self.metodo.calcola_previsione(self.archivio.dati_per_analisi[data_calcolo], fisso=fisso)
            if not previsione: continue
            self.ultimo_caso_analizzato = previsione
            esito = self._controlla_esito(previsione, data_calcolo, self.backtest_params['colpi_da_giocare'])
            if not silent:
                self._log(f"\n{'='*60}\nData Calcolo: {data_calcolo}")
                self._stampa_previsione(previsione, show_header=False)
                if esito['vinto']:
                    testo_esito = f"VINTO! {esito['tipo']} al colpo {esito['colpo']} su {esito['ruota']} ({esito['numeri_vincenti']})"
                else:
                    testo_esito = f"NEGATIVO (nessuna vincita nei {self.backtest_params['colpi_da_giocare']} colpi)"
                self._log(f"--> ESITO: {testo_esito}")
            
            if esito['vinto']: vincite[esito['tipo']] += 1
            elif fisso == 72:
                numeri_sorgente = previsione['numeri_sorgente']
                esiti_reali = self._trova_esiti_reali(data_calcolo, self.backtest_params['colpi_da_giocare'], previsione['ruote'])
                self.casi_negativi_per_correttore.append({'terzo_na': numeri_sorgente['NA']['3'], 'terzo_pa': numeri_sorgente['PA']['3'], 'esiti_reali': esiti_reali})
        
        risultati_finali = {'totale_previsioni': len(date_da_analizzare), 'vincite_ambata': vincite['Ambata'], 'vincite_ambo': vincite['Ambo'], 'vincite_terno': vincite['Terno']}
        if not silent: self._stampa_riepilogo_backtest(risultati_finali)
        return risultati_finali

    def _stampa_riepilogo_backtest(self, risultati):
        self._log(f"\n\n{'='*60}\n--- RIEPILOGO FINALE BACKTEST ---\n")
        tot_previsioni = risultati['totale_previsioni']
        v_ambata, v_ambo, v_terno = risultati['vincite_ambata'], risultati['vincite_ambo'], risultati['vincite_terno']
        self._log(f"Previsioni totali calcolate: {tot_previsioni}")
        if tot_previsioni > 0:
            tot_vincite_ambata = v_ambata + v_ambo + v_terno
            self._log(f"Vincite per Ambata: {tot_vincite_ambata} ({(tot_vincite_ambata / tot_previsioni) * 100:.2f}%)")
            tot_vincite_ambo = v_ambo + v_terno
            self._log(f"Vincite per Ambo:   {tot_vincite_ambo} ({(tot_vincite_ambo / tot_previsioni) * 100:.2f}%)")
            self.output_queue.put(f"Vincite per Terno:  {v_terno} ({(v_terno / tot_previsioni) * 100:.2f}%)")
        else:
            self._log("Vincite per Ambata: 0"); self._log("Vincite per Ambo:   0"); self.output_queue.put("Vincite per Terno:  0")
        self.output_queue.put(f"\n{'='*60}")

    def _controlla_esito(self, previsione, data_calcolo, max_colpi):
        ambata, (ab1, ab2) = previsione['ambata'], previsione['abbinamenti']
        ambo1, ambo2, terno = {ambata, ab1}, {ambata, ab2}, {ambata, ab1, ab2}
        start_index = self.archivio.date_to_index.get(data_calcolo, -1) + 1
        for colpo in range(1, max_colpi + 1):
            current_index = start_index + colpo - 1
            if current_index >= len(self.archivio.date_ordinate): break
            estrazione = self.archivio.dati_per_analisi[self.archivio.date_ordinate[current_index]]
            for ruota in previsione['ruote']:
                numeri_estratti = set(estrazione.get(ruota, []))
                if terno.issubset(numeri_estratti): return {'vinto': True, 'tipo': 'Terno', 'colpo': colpo, 'ruota': ruota, 'numeri_vincenti': sorted(list(terno))}
                if ambo1.issubset(numeri_estratti) or ambo2.issubset(numeri_estratti): return {'vinto': True, 'tipo': 'Ambo', 'colpo': colpo, 'ruota': ruota, 'numeri_vincenti': sorted(list(ambo1 if ambo1.issubset(numeri_estratti) else ambo2))}
                if ambata in numeri_estratti: return {'vinto': True, 'tipo': 'Ambata', 'colpo': colpo, 'ruota': ruota, 'numeri_vincenti': [ambata]}
        return {'vinto': False}

    def _trova_esiti_reali(self, data_calcolo, max_colpi, ruote_gioco):
        esiti_reali, start_index = set(), self.archivio.date_to_index.get(data_calcolo, -1) + 1
        for colpo in range(1, max_colpi + 1):
            current_index = start_index + colpo - 1
            if current_index >= len(self.archivio.date_ordinate): break
            estrazione = self.archivio.dati_per_analisi[self.archivio.date_ordinate[current_index]]
            for ruota in ruote_gioco:
                if numeri_estratti := estrazione.get(ruota, []): esiti_reali.update(numeri_estratti)
        return esiti_reali

    def _run_corrector(self): self._clear_output(); self._run_in_thread(self._corrector_task)
    def _corrector_task(self):
        if not self.casi_negativi_per_correttore or not self.ultimo_caso_analizzato:
            self._log("Esegui prima un'analisi storica completa."); return
        self._log(f"--- ANALISI CORRETTORI PER AMBATA ---\n\nBasato su {len(self.casi_negativi_per_correttore)} esiti negativi.\nFormula testata: (terzo_na + terzo_pa + [CORRETTORE])\n")
        risultati = Correttore().trova_correttore_somma(self.casi_negativi_per_correttore)
        self._log("--- CLASSIFICA MIGLIORI CORRETTORI (FISSI DA SOMMARE) ---\n")
        if not risultati: self._log("Nessun correttore è riuscito a migliorare gli esiti."); return
        self.miglior_correttore_trovato = risultati[0][0]
        self.validator_button.config(text=f"Valida Miglior Correttore (+{self.miglior_correttore_trovato})", state="normal")
        src, (ab1, ab2) = self.ultimo_caso_analizzato['numeri_sorgente'], self.ultimo_caso_analizzato['abbinamenti']
        for i, (correttore_val, successi) in enumerate(risultati[:15]):
            perc = (successi / len(self.casi_negativi_per_correttore)) * 100
            nuova_ambata = self.metodo._fuori_90(src['NA']['3'] + src['PA']['3'] + correttore_val)
            self._log(f"{i+1}. Correttore +{correttore_val:<2}:  (Successi: {successi} | {perc:.2f}%)")
            self._log(f"   -> Nuova Previsione: Ambata {nuova_ambata}, Terzina {nuova_ambata}-{ab1}-{ab2}\n" + "-" * 65)

    def _run_validator(self): self._clear_output(); self._run_in_thread(self._validator_task)
    def _validator_task(self):
        if self.miglior_correttore_trovato is None: self._log("Trova prima un correttore ottimale."); return
        self._log(f"--- VALIDAZIONE GLOBALE DEL CORRETTORE +{self.miglior_correttore_trovato} ---\nPeriodo: {self.backtest_params['start_date_str']} / {self.backtest_params['end_date_str']}\n")
        self._log("\n" + "="*50 + "\n  METODO ORIGINALE (con fisso +72)\n" + "="*50)
        res_orig = self._esegui_backtest_concreto(fisso=72, silent=True)
        self._stampa_riepilogo_backtest(res_orig)
        self._log("\n" + "="*50 + f"\n  METODO OTTIMIZZATO (con fisso +{self.miglior_correttore_trovato})\n" + "="*50)
        res_opt = self._esegui_backtest_concreto(fisso=self.miglior_correttore_trovato, silent=True)
        self._stampa_riepilogo_backtest(res_opt)
        self._log("\n" + "="*50 + "\n  BILANCIO FINALE\n" + "="*50)
        self._log(f"Variazione Ambata: {res_opt['vincite_ambata'] - res_orig['vincite_ambata']:+}")
        self._log(f"Variazione Ambo:   {res_opt['vincite_ambo'] - res_orig['vincite_ambo']:+}")
        self._log(f"Variazione Terno:  {res_opt['vincite_terno'] - res_orig['vincite_terno']:+}")
        tot_orig = res_orig['vincite_ambata'] + res_orig['vincite_ambo'] + res_orig['vincite_terno']
        tot_opt = res_opt['vincite_ambata'] + res_opt['vincite_ambo'] + res_opt['vincite_terno']
        self._log("-" * 25 + f"\nGuadagno Netto Totale: {tot_opt - tot_orig:+} previsioni vincenti.")

    def _stampa_previsione(self, previsione, show_header=True):
        if show_header: self._log("\n--- PREVISIONE GENERATA ---")
        src, (ambata, (ab1, ab2)) = previsione['numeri_sorgente'], (previsione['ambata'], previsione['abbinamenti'])
        self._log(f"(Calcolo basato su NA: {src['NA']['3']},{src['NA']['5']} e PA: {src['PA']['3']},{src['PA']['5']})")
        self._log(f"  Ruote di Gioco: Napoli - Palermo\n  Ambata: {ambata}\n  Ambi: {ambata}-{ab1}  e  {ambata}-{ab2}\n  Terzina: {ambata}-{ab1}-{ab2}")
        if show_header: self._log("---------------------------\n")

    def _process_queue(self):
        try:
            while True:
                msg = self.output_queue.get_nowait(); self.output_text.config(state='normal'); self.output_text.insert(tk.END, msg + "\n"); self.output_text.see(tk.END); self.output_text.config(state='disabled')
        except queue.Empty: pass
        finally: self.after(100, self._process_queue)

if __name__ == "__main__":
    app = MetodoApp()
    app.mainloop()