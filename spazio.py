import numpy as np
import pandas as pd
import requests
import itertools
from collections import defaultdict

# ==============================================================================
# 1. CONFIGURAZIONE E FUNZIONI DI UTILITÀ GLOBALI
# ==============================================================================

GITHUB_USER = "illottodimax"
GITHUB_REPO = "Archivio"
GITHUB_BRANCH = "main"

RUOTE_MAP = {
    'BARI': 'BA', 'CAGLIARI': 'CA', 'FIRENZE': 'FI', 'GENOVA': 'GE',
    'MILANO': 'MI', 'NAPOLI': 'NA', 'PALERMO': 'PA', 'ROMA': 'RO',
    'TORINO': 'TO', 'VENEZIA': 'VE', 'NAZIONALE': 'NZ'
}

def get_coords_map():
    coords_map = {}
    for i in range(90): coords_map[i + 1] = (i // 10, i % 10)
    return coords_map

COORDS_MAP = get_coords_map()

def carica_estrazioni_da_github(nome_ruota, num_estrazioni=None):
    nome_file = f"{nome_ruota.upper()}.txt"
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{nome_file}"
    print(f"[INFO] Tentativo di caricamento dati per {nome_ruota.upper()}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERRORE] Impossibile caricare il file da GitHub. Dettagli: {e}"); return []
    estrazioni_trovate = []
    for ln, riga in enumerate(response.text.strip().split('\n'), 1):
        parti = riga.strip().split()
        if len(parti) == 7:
            try: estrazioni_trovate.append((parti[0], [int(n) for n in parti[2:7]]))
            except ValueError: print(f"[AVVISO] Riga {ln} ignorata (formato non valido): '{riga}'")
    if not estrazioni_trovate: print("[AVVISO] Nessuna estrazione valida trovata."); return []
    return estrazioni_trovate[-num_estrazioni:] if num_estrazioni else estrazioni_trovate

# ==============================================================================
# 2. CLASSE PRINCIPALE PER L'ANALISI SPAZIOMETRICA
# ==============================================================================

class SpaziometriaLotto:
    def __init__(self):
        self.date_estrazioni, self.estrazioni_originali, self.estrazioni = [], [], []
        self.distanze, self.statistiche_distanze = [], defaultdict(int)
        self.nome_ruota_analizzata = "N/D"

    def carica_estrazioni(self, estrazioni_con_data, nome_ruota):
        if not estrazioni_con_data: return
        self.date_estrazioni = [e[0] for e in estrazioni_con_data]
        self.estrazioni_originali = [e[1] for e in estrazioni_con_data]
        self.estrazioni = [sorted(e) for e in self.estrazioni_originali]
        self.distanze = [self._calcola_distanze(e) for e in self.estrazioni]
        self.statistiche_distanze = defaultdict(int)
        for d_list in self.distanze:
            for d in d_list: self.statistiche_distanze[d] += 1
        self.nome_ruota_analizzata = nome_ruota
        
    def _calcola_distanze(self, numeri_ordinati):
        return [numeri_ordinati[i+1] - numeri_ordinati[i] for i in range(len(numeri_ordinati)-1)]

    def _applica_figura(self, primo_estratto, figura):
        previsione, n_corr = [primo_estratto], primo_estratto
        for d in figura:
            n_corr += d
            previsione.append(n_corr % 90 if n_corr > 90 else n_corr)
        return sorted(list(set(previsione)))

    def analisi_distanze_base(self):
        if not self.distanze: return
        tutte_distanze = [d for estrazione in self.distanze for d in estrazione]
        print(f"\n--- ANALISI STATISTICA - RUOTA DI {self.nome_ruota_analizzata.upper()} ---")
        print(f"Numero estrazioni analizzate: {len(self.estrazioni)}")
        print(f"Distanza media: {np.mean(tutte_distanze):.2f}, Dev. standard: {np.std(tutte_distanze):.2f}")

    def frequenza_distanze(self, top_n=15):
        if not self.statistiche_distanze: return
        print(f"\n--- TOP {top_n} DISTANZE PIÙ FREQUENTI ---")
        distanze_ordinate = sorted(self.statistiche_distanze.items(), key=lambda x: x[1], reverse=True)
        total_freq = sum(self.statistiche_distanze.values())
        for i, (dist, freq) in enumerate(distanze_ordinate[:top_n]):
            perc = (freq / total_freq) * 100
            print(f"{i+1:2d}. Distanza {dist:2d}: {freq:4d} volte ({perc:.2f}%)")

    def _calcola_dist_sq(self, p1, p2): return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

    def _get_figure_type(self, p1, p2, p3, p4):
        if (p1[0] + p3[0] == p2[0] + p4[0]) and (p1[1] + p3[1] == p2[1] + p4[1]):
            d1_sq,d2_sq,l1_sq = self._calcola_dist_sq(p1,p3),self._calcola_dist_sq(p2,p4),self._calcola_dist_sq(p1,p2)
            is_r, is_rh = abs(d1_sq-d2_sq)<1e-9, abs(d1_sq+d2_sq-4*l1_sq)<1e-9
            if is_r and is_rh: return "QUADRATO"
            if is_r: return "RETTANGOLO"
            if is_rh: return "ROMBO"
            return "PARALLELOGRAMMA"
        return None

    def analisi_figure_geometriche(self, analizza_tutto_archivio=True):
        print("\n--- ANALISI FIGURE GEOMETRICHE COMPLETE NELL'ARCHIVIO ---")
        est_an, date_an = (self.estrazioni, self.date_estrazioni) if analizza_tutto_archivio else (self.estrazioni[-10:], self.date_estrazioni[-10:])
        print(f"Scansione di {len(est_an)} estrazioni...")
        fig_trovate, log = defaultdict(int), []
        for i, estr in enumerate(est_an):
            trovate_in_estr = set()
            for combo in itertools.combinations(estr, 4):
                combo_t = tuple(sorted(combo))
                if combo_t in trovate_in_estr: continue
                p = [COORDS_MAP[n] for n in combo]; p1, p2, p3, p4 = p[0], p[1], p[2], p[3]
                if (p1[0]+p4[0]==p2[0]+p3[0]) and (p1[1]+p4[1]==p2[1]+p3[1]):
                    tipo=self._get_figure_type(p1,p2,p4,p3)
                    if tipo: log.append(f"  - {date_an[i]}: Trovato {tipo} con {sorted(combo)}"); fig_trovate[tipo]+=1; trovate_in_estr.add(combo_t); continue
                if (p1[0]+p3[0]==p2[0]+p4[0]) and (p1[1]+p3[1]==p2[1]+p4[1]):
                    tipo=self._get_figure_type(p1,p2,p3,p4)
                    if tipo: log.append(f"  - {date_an[i]}: Trovato {tipo} con {sorted(combo)}"); fig_trovate[tipo]+=1; trovate_in_estr.add(combo_t); continue
                if (p1[0]+p2[0]==p3[0]+p4[0]) and (p1[1]+p2[1]==p3[1]+p4[1]):
                    tipo=self._get_figure_type(p1,p3,p2,p4)
                    if tipo: log.append(f"  - {date_an[i]}: Trovato {tipo} con {sorted(combo)}"); fig_trovate[tipo]+=1; trovate_in_estr.add(combo_t)
        if not log: print("Nessuna figura notevole trovata.")
        else:
            print("\nEventi geometrici rilevati:"); [print(e) for e in log]; print("\n  Riepilogo Affidabilità Geometrica:")
            for fig, count in sorted(fig_trovate.items()): print(f"    - {fig}: {count} volte")

    def previsione_da_completamento_figura(self):
        print("\n--- Previsione da Completamento Figura Geometrica (Parallelogrammi) ---")
        if not self.estrazioni: return []
        print(f"Previsione valida dall'estrazione successiva al: {self.date_estrazioni[-1]}")
        previsioni = self._previsione_geometrica_silenziosa()
        if previsioni:
            print(f"Basandosi sull'ultima estrazione {self.estrazioni[-1]}:"); print(f"Numeri per completare PARALLELOGRAMMI: {previsioni}")
            print("Si consiglia di giocarli come AMBATA o AMBO su ruota e Tutte.")
        else: print("Nessuna figura (parallelogramma) semplice da completare trovata.")
        return previsioni
            
    def previsione_da_figura_frequente(self, top_n=1):
        if not self.distanze: return []
        figure_freq = defaultdict(int)
        for d in self.distanze:
            figure_freq[tuple(d)] += 1
        if not figure_freq: return []
        figure_ordinate = sorted(figure_freq.items(), key=lambda x: x[1], reverse=True)
        if not figure_ordinate: return []
        figura, frequenza = figure_ordinate[top_n-1]
        primo, data = self.estrazioni_originali[-1][0], self.date_estrazioni[-1]
        previsione = self._applica_figura(primo, figura)
        print(f"\n--- Previsione da Figura Più Frequente (la N.{top_n}) ---")
        print(f"Previsione valida dall'estrazione successiva al: {data}")
        print(f"Figura più frequente: {figura} (uscita {frequenza} volte in totale)")
        print(f"Applicata al 1° estratto dell'ultima estrazione ({primo})"); print(f"PREVISIONE CINQUINA: {previsione}")
        if frequenza > 1: 
            self.verifica_storia_figura(figura)
        return previsione
            
    def verifica_storia_figura(self, figura_da_cercare, colpi_di_gioco=5):
        print("\n" + "#"*60); print(f"  VERIFICA STORICA DELLA FIGURA: {figura_da_cercare}"); print("#"*60)
        occorrenze = [i for i, d in enumerate(self.distanze) if tuple(d) == figura_da_cercare]
        occorrenze_passate = occorrenze[:-1] 
        if not occorrenze_passate: print("Questa figura non ha uno storico di ripetizioni passate da analizzare."); return
        print(f"La figura si è ripetuta {len(occorrenze_passate)} volte in passato. Analisi storica:\n")
        for i in occorrenze_passate:
            data_ev, primo_ev = self.date_estrazioni[i], self.estrazioni_originali[i][0]
            prev_gen = self._applica_figura(primo_ev, figura_da_cercare)
            print(f"--- Occorrenza del {data_ev} ---"); print(f"    Primo estratto: {primo_ev}"); print(f"    Previsione generata: {prev_gen}")
            esito = False
            for colpo in range(colpi_di_gioco):
                idx = i + 1 + colpo
                if idx < len(self.estrazioni_originali):
                    data_r, estr_r = self.date_estrazioni[idx], self.estrazioni_originali[idx]
                    vincenti = set(prev_gen).intersection(set(estr_r))
                    if vincenti:
                        tipo = "AMBATA" if len(vincenti)==1 else "AMBO" if len(vincenti)==2 else "TERNO+"
                        print(f"    --> ESITO: POSITIVO! Vinto al colpo {colpo+1} ({data_r}) con {tipo} {list(vincenti)}")
                        esito = True; break
            if not esito: print(f"    --> ESITO: NEGATIVO. Nessuna vincita in {colpi_di_gioco} colpi.")
            print("-" * 20)

    def _previsione_geometrica_silenziosa(self, return_all_candidates=False):
        if len(self.estrazioni) < 1: return []
        previsioni = [] if return_all_candidates else set()
        ultima_estr = self.estrazioni[-1]
        for combo in itertools.combinations(ultima_estr, 3):
            p1, p2, p3 = (COORDS_MAP[n] for n in combo)
            punti = [(p1, p2, p3), (p2, p3, p1), (p1, p3, p2)]
            for A, B, C in punti:
                px4, py4 = B[0] + C[0] - A[0], B[1] + C[1] - A[1]
                if 0 <= px4 < 9 and 0 <= py4 < 10:
                    n_prev = int(px4 * 10 + py4 + 1)
                    if 1 <= n_prev <= 90 and n_prev not in ultima_estr:
                        previsioni.append(n_prev) if return_all_candidates else previsioni.add(n_prev)
        return previsioni if return_all_candidates else sorted(list(previsioni))

    def _previsione_frequenza_silenziosa(self, top_n=1):
        if not self.distanze: return []
        figure_freq = defaultdict(int)
        for d in self.distanze: figure_freq[tuple(d)] += 1
        if not figure_freq: return []
        figure_ordinate = sorted(figure_freq.items(), key=lambda x: x[1], reverse=True)
        if not figure_ordinate: return []
        figura_top = figure_ordinate[top_n-1][0]
        primo_estratto = self.estrazioni_originali[-1][0]
        return self._applica_figura(primo_estratto, figura_top)

    def esegui_backtest(self, metodo, colpi_di_gioco=5, start_offset=20):
        print("\n" + "#"*60)
        print(f"  AVVIO BACKTEST - RUOTA: {self.nome_ruota_analizzata.upper()} - METODO: '{metodo.upper()}' - COLPI: {colpi_di_gioco}")
        print("#"*60)
        
        if len(self.estrazioni_originali) < start_offset + colpi_di_gioco:
            print(f"[ERRORE] Dati insufficienti. Servono almeno {start_offset + colpi_di_gioco} estrazioni per un test significativo."); return
        
        vincite, risultati, n_previsioni = 0, [], 0
        previsioni_in_gioco = []

        for i in range(start_offset, len(self.estrazioni_originali)):
            dati_storici = list(zip(self.date_estrazioni[:i], self.estrazioni_originali[:i]))
            
            analizzatore_storico = SpaziometriaLotto()
            analizzatore_storico.carica_estrazioni(dati_storici, self.nome_ruota_analizzata)
            
            numeri_previsti = []
            if metodo == 'geometrico':
                tutti_suggerimenti = analizzatore_storico._previsione_geometrica_silenziosa(return_all_candidates=True)
                if tutti_suggerimenti:
                    conteggi = defaultdict(int)
                    for n in tutti_suggerimenti: conteggi[n] += 1
                    if conteggi:
                        max_freq = max(conteggi.values())
                        numeri_previsti = sorted([n for n, freq in conteggi.items() if freq == max_freq])
            else:
                numeri_previsti = analizzatore_storico._previsione_frequenza_silenziosa()

            if not numeri_previsti: continue
            
            n_previsioni += 1
            esito_trovato = False
            
            for colpo in range(colpi_di_gioco):
                idx_futuro = i + colpo
                if idx_futuro < len(self.estrazioni_originali):
                    estr_reale, data_reale = self.estrazioni_originali[idx_futuro], self.date_estrazioni[idx_futuro]
                    vincenti = set(numeri_previsti).intersection(set(estr_reale))
                    if vincenti:
                        vincite += 1
                        tipo = "AMBATA" if len(vincenti) == 1 else "AMBO" if len(vincenti) == 2 else "TERNO+"
                        risultati.append(f"VINCITA colpo {colpo+1}! Prev. del {dati_storici[-1][0]} ({numeri_previsti}) -> Vinto il {data_reale} con {tipo} {list(vincenti)}")
                        esito_trovato = True
                        break
            
            if not esito_trovato:
                colpi_trascorsi = len(self.estrazioni_originali) - i
                
                if colpi_trascorsi < colpi_di_gioco:
                    colpi_rimanenti = colpi_di_gioco - colpi_trascorsi
                    previsioni_in_gioco.append({
                        "data_generazione": dati_storici[-1][0],
                        "numeri": numeri_previsti,
                        "colpi_trascorsi": colpi_trascorsi,
                        "colpi_rimanenti": colpi_rimanenti
                    })
                else:
                    risultati.append(f"NEGATIVO. Prev. del {dati_storici[-1][0]} ({numeri_previsti}) non uscita in {colpi_di_gioco} colpi.")

        print("\n--- RISULTATI DETTAGLIATI BACKTEST (EVENTI CONCLUSI) ---\n")
        if risultati:
            [print(res) for res in risultati]
        else:
            print("Nessuna previsione conclusa da mostrare.")
        
        print(f"\n--- RIEPILOGO BACKTEST (EVENTI CONCLUSI) - RUOTA DI {self.nome_ruota_analizzata.upper()} ---")
        previsioni_concluse = n_previsioni - len(previsioni_in_gioco)
        if previsioni_concluse > 0:
            perc_successo = (vincite / previsioni_concluse) * 100
            print(f"Strategia: '{metodo.upper()}'")
            print(f"Previsioni Concluse: {previsioni_concluse} | Vincite: {vincite} | Successo: {perc_successo:.2f}%")
        else:
            print("Nessuna previsione conclusa per calcolare un riepilogo statistico.")
            
        print("\n" + "="*60)
        print(f"--- PREVISIONI ANCORA IN GIOCO SU RUOTA DI {self.nome_ruota_analizzata.upper()} ---")
        print(f"Analisi basata sulle ultime estrazioni e {colpi_di_gioco} colpi di gioco.")
        print("="*60)

        if not previsioni_in_gioco:
            print("\nNessuna previsione risulta ancora attiva per questa ruota.")
        else:
            print("\nLe seguenti previsioni non hanno ancora dato esito e sono considerate ATTIVE:")
            for p in sorted(previsioni_in_gioco, key=lambda x: x['data_generazione'], reverse=True):
                print("-" * 35)
                print(f"> Previsione generata il: {p['data_generazione']}")
                
                numeri_gioco = p['numeri']
                num_count = len(numeri_gioco)
                
                if num_count == 1:
                    print(f"  Suggerimento: AMBATA SECCA")
                    print(f"  Numero: {numeri_gioco[0]}")
                elif num_count == 2:
                    print(f"  Suggerimento: AMBO SECCO")
                    print(f"  Numeri: {numeri_gioco}")
                    print(f"  (Alternativa: giocare {numeri_gioco[0]} e {numeri_gioco[1]} come ambate separate)")
                else:
                    print(f"  Suggerimento: LUNGHETTA per AMBO/TERNO")
                    print(f"  Numeri: {numeri_gioco}")
                    print(f"  Strategia CAPOGIOCO: {numeri_gioco[0]} come Ambata,")
                    print(f"  e in abbinamento per Ambo con {numeri_gioco[1:]}")

                print(f"\n  Stato: {p['colpi_trascorsi']}/{colpi_di_gioco} colpi trascorsi. Colpi rimanenti: {p['colpi_rimanenti']}.")
        print("-" * 35)

# ==============================================================================
# 3. BLOCCO DI ESECUZIONE PRINCIPALE
# ==============================================================================
if __name__ == "__main__":
    while True:
        print("\n" + "="*50); print("--- ANALIZZATORE SPAZIOMETRICO LOTTO (by ilLottoDiMax) ---"); print("="*50)
        print("Scegli un'opzione:")
        ruote_elenco = list(RUOTE_MAP.keys())
        for i, nome_ruota in enumerate(ruote_elenco): print(f"{i+1:2d}. Analizza Ruota di {nome_ruota}")
        print("-----------------------------------------"); print("98. Esegui Backtest di una strategia"); print("99. Esci dal programma")
        try:
            scelta_principale = int(input("\n> Inserisci il numero della tua scelta: "))
        except ValueError:
            print("[ERRORE] Input non valido. Inserisci un numero."); continue
            
        if scelta_principale == 99: print("Uscita dal programma. Arrivederci!"); break
        
        elif scelta_principale == 98:
            print("\n--- Modalità Backtest ---")
            try:
                ruota_idx_input = input(f"Su quale ruota vuoi testare? (1-{len(ruote_elenco)}, INVIO per tutte): ")
                ruote_da_testare = []
                if not ruota_idx_input:
                    ruote_da_testare = ruote_elenco
                    print("[INFO] Verrà eseguito il backtest su TUTTE le ruote.")
                else:
                    ruota_idx = int(ruota_idx_input) - 1
                    if not 0 <= ruota_idx < len(ruote_elenco): 
                        print("[ERRORE] Scelta ruota non valida."); continue
                    ruote_da_testare.append(ruote_elenco[ruota_idx])

                print("\nQuale strategia vuoi testare?"); print("1. Completamento Figura Geometrica"); print("2. Figura Spaziometrica più Frequente")
                metodo_idx = int(input("> Scelta strategia (1 o 2): "))
                metodo_test = 'geometrico' if metodo_idx == 1 else 'frequenza' if metodo_idx == 2 else None
                if metodo_test is None: print("[ERRORE] Scelta strategia non valida."); continue
                
                while True:
                    try:
                        colpi_input = input("> Su quanti colpi vuoi verificare ogni previsione? (es. 18): ")
                        colpi = int(colpi_input)
                        if colpi > 0: break
                        else: print("[ERRORE] Il numero di colpi deve essere maggiore di zero. Riprova.")
                    except ValueError: print("[ERRORE] Input non valido. Devi inserire un numero intero. Riprova.")

                # === NUOVA DOMANDA PER IL BACKTEST ===
                while True:
                    try:
                        num_test_input = input("> Su quante delle ultime estrazioni vuoi eseguire il test? (es. 50, INVIO per tutto l'archivio): ")
                        num_test_da_eseguire = int(num_test_input) if num_test_input else None
                        if num_test_da_eseguire is None or num_test_da_eseguire > 0: break
                        else: print("[ERRORE] Il numero di test deve essere maggiore di zero. Riprova.")
                    except ValueError: print("[ERRORE] Input non valido. Devi inserire un numero intero. Riprova.")
                # === FINE NUOVA DOMANDA ===

                for ruota_test in ruote_da_testare:
                    print(f"\n{'='*10} INIZIO TEST SU RUOTA DI {ruota_test.upper()} {'='*10}")
                    archivio = carica_estrazioni_da_github(ruota_test)
                    if archivio:
                        # === CALCOLO DEL PUNTO DI PARTENZA PER IL TEST ===
                        start_offset_test = 20 # Valore minimo di default
                        if num_test_da_eseguire is not None:
                            # Calcola da dove iniziare a testare per avere il numero di test richiesto
                            start_index = len(archivio) - num_test_da_eseguire
                            # Assicurati che ci sia comunque una cronologia minima per il primo test
                            start_offset_test = max(20, start_index)
                            print(f"[INFO] Verranno analizzate le ultime {len(archivio) - start_offset_test} occasioni di previsione.")
                        else:
                            print("[INFO] Verrà analizzato l'intero archivio disponibile.")
                        
                        analizzatore = SpaziometriaLotto()
                        analizzatore.carica_estrazioni(archivio, ruota_test)
                        # Passiamo il nuovo start_offset calcolato
                        analizzatore.esegui_backtest(metodo=metodo_test, colpi_di_gioco=colpi, start_offset=start_offset_test)
                    else: 
                        print(f"[ERRORE] Impossibile caricare l'archivio per la ruota di {ruota_test}.")

            except ValueError: print("[ERRORE] Input non valido durante la configurazione del backtest. Riprova.")
            input("\nPremi INVIO per tornare al menu principale..."); continue

        elif 1 <= scelta_principale <= len(ruote_elenco):
            RUOTA_SCELTA = ruote_elenco[scelta_principale - 1]
            try:
                inp_estr = input(f"> Quante estrazioni di {RUOTA_SCELTA} vuoi analizzare? (INVIO per tutte): ")
                NUM_ESTRAZIONI = int(inp_estr) if inp_estr else None
            except ValueError: print("[INFO] Input non valido. Analizzo tutte."); NUM_ESTRAZIONI = None
            estrazioni_caricate = carica_estrazioni_da_github(RUOTA_SCELTA, NUM_ESTRAZIONI)
            if estrazioni_caricate:
                msg = f"le ultime {len(estrazioni_caricate)}" if NUM_ESTRAZIONI else "tutte le"
                print(f"\n[INFO] Caricate {msg} estrazioni per la ruota di {RUOTA_SCELTA.upper()}.")
                spaziometria = SpaziometriaLotto()
                spaziometria.carica_estrazioni(estrazioni_caricate, RUOTA_SCELTA)
                spaziometria.analisi_distanze_base(); spaziometria.frequenza_distanze()
                print("\n\n" + "*"*40); print("    ANALISI E PREVISIONI GEOMETRICHE"); print("*"*40)
                spaziometria.analisi_figure_geometriche()
                spaziometria.previsione_da_completamento_figura()
                print("\n\n" + "*"*40); print("    PREVISIONI SPAZIOMETRICHE (Distanze)"); print("*"*40)
                spaziometria.previsione_da_figura_frequente()
                print("\nAVVERTENZA: Le previsioni sono generate a scopo di studio e non garantiscono alcuna vincita.")
            else: print(f"\n[ERRORE] Analisi fallita per la ruota di {RUOTA_SCELTA}.")
            input("\nPremi INVIO per continuare e scegliere un'altra ruota...")
            
        else: print("\n[ERRORE] Scelta non valida. Riprova.")