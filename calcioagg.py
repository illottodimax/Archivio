from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re # Per espressioni regolari, utili per pulire testo
import os 
import random # Per pause randomizzate

# --- CONFIGURAZIONE CAMPIONATI ---
CAMPIΟNATI_DA_SCRAPARE = [
    {
        "nome_visualizzato": "Serie A (Italia)",
        "url_principale_recente": "https://www.livescore.in/it/calcio/italia/serie-a/#/zDpS37lb/table/overall", # URL RIEPILOGO
        "url_storico_aggiuntivo": "https://www.livescore.in/it/calcio/italia/serie-a/risultati/",          # Pagina Risultati per lo storico
        "url_calendario_esteso": "https://www.livescore.in/it/calcio/italia/serie-a/calendario/",         # Pagina Calendario per futuro esteso
        "codice_divisione": "I1",
    },
    {
        "nome_visualizzato": "Serie B (Italia)",
        "url_principale_recente": "https://www.livescore.in/it/calcio/italia/serie-b/#/dSYz2oJA/draw",       # CHIAVE CORRETTA
        "url_storico_aggiuntivo": "https://www.livescore.in/it/calcio/italia/serie-b/risultati/",          # CHIAVE CORRETTA
        "url_calendario_esteso": "https://www.livescore.in/it/calcio/italia/serie-b/calendario/",         # CHIAVE CORRETTA
        "codice_divisione": "I2", # VIRGOLA AGGIUNTA
    },
    {
        "nome_visualizzato": "La Liga (Spagna)",
        "url_principale_recente": "https://www.livescore.in/it/calcio/spagna/laliga/#/dINOZk9Q/table/overall", # CHIAVE CORRETTA
        "url_storico_aggiuntivo": "https://www.livescore.in/it/calcio/spagna/laliga/risultati/",          # CHIAVE CORRETTA
        "url_calendario_esteso": "https://www.livescore.in/it/calcio/spagna/laliga/calendario/",         # CHIAVE CORRETTA
        "codice_divisione": "SP1",
    },
    {
        "nome_visualizzato": "Ligue 1 (Francia)",
        "url_principale_recente": "https://www.livescore.in/it/calcio/francia/ligue-1/#/U16IzNAd/draw",    # CHIAVE CORRETTA
        "url_storico_aggiuntivo": "https://www.livescore.in/it/calcio/francia/ligue-1/risultati/",      # CHIAVE CORRETTA
        "url_calendario_esteso": "https://www.livescore.in/it/calcio/francia/ligue-1/calendario/",     # CHIAVE CORRETTA
        "codice_divisione": "F1",
    },
    {
        "nome_visualizzato": "Bundesliga (Germania)",
        "url_principale_recente": "https://www.livescore.in/it/calcio/germania/bundesliga/#/Uc2VcOR5/draw", # CHIAVE CORRETTA
        "url_storico_aggiuntivo": "https://www.livescore.in/it/calcio/germania/bundesliga/risultati/",  # CHIAVE CORRETTA
        "url_calendario_esteso": "https://www.livescore.in/it/calcio/germania/bundesliga/calendario/", # CHIAVE CORRETTA
        "codice_divisione": "D1",
    },
    {
        "nome_visualizzato": "Premier League (Inghilterra)",
        "url_principale_recente": "https://www.livescore.in/it/calcio/inghilterra/premier-league/#/lAkHuyP3/table/overall", # CHIAVE CORRETTA
        "url_storico_aggiuntivo": "https://www.livescore.in/it/calcio/inghilterra/premier-league/risultati/",              # CHIAVE CORRETTA
        "url_calendario_esteso": "https://www.livescore.in/it/calcio/inghilterra/premier-league/calendario/",             # CHIAVE CORRETTA
        "codice_divisione": "E0",
    },
    {
        "nome_visualizzato": "African Nation Championship",
        "url_principale_recente": "https://www.livescore.in/it/calcio/africa/african-nations-championship/#/2kJ46vuf/table", # CHIAVE CORRETTA
        "url_storico_aggiuntivo": "https://www.livescore.in/it/calcio/africa/african-nations-championship/risultati/",      # CHIAVE CORRETTA e https:// corretto
        "url_calendario_esteso": "https://www.livescore.in/it/calcio/africa/african-nations-championship/calendario/",     # CHIAVE CORRETTA
        "codice_divisione": "ANC", 
    },
]

OUTPUT_CSV_GLOBALE = "dati_livescore_tutti_campionati.csv"

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--log-level=3") 
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        return driver
    except Exception as e:
        print(f"Errore durante l'inizializzazione del WebDriver: {e}")
        return None

def parse_date_flashscore(date_str_raw, year_context):
    if not date_str_raw: return None
    date_str_cleaned = re.sub(r'(Oggi|Ieri|Dom|Lun|Mar|Mer|Gio|Ven|Sab)\s*', '', date_str_raw, flags=re.IGNORECASE).strip()
    date_part_match = re.match(r'(\d{1,2}\.\d{1,2}\.(\d{4})?|\d{1,2}\.\d{1,2}\.?)', date_str_cleaned)
    if not date_part_match: return None
    date_str = date_part_match.group(0).strip().rstrip('.')
    try:
        parts = date_str.split('.')
        if len(parts) == 3 and parts[2]: dt_obj = datetime.strptime(date_str, "%d.%m.%Y")
        elif len(parts) == 2:
            day, month = int(parts[0]), int(parts[1])
            current_dt = datetime.now()
            if month > current_dt.month and (current_dt.month < 3 or month > 10): # Heuristica per cambio anno
                 dt_obj_candidate = datetime(year_context -1, month, day)
                 dt_obj = dt_obj_candidate if (current_dt - dt_obj_candidate).days <= 180 else datetime(year_context, month, day)
            elif month < current_dt.month and (current_dt.month > 10 or month < 3): # Heuristica per cambio anno
                 dt_obj_candidate = datetime(year_context +1, month, day)
                 dt_obj = dt_obj_candidate if (dt_obj_candidate - current_dt).days <= 180 else datetime(year_context, month, day)
            else: dt_obj = datetime(year_context, month, day)
        else: return None
        return dt_obj.strftime("%d/%m/%y")
    except ValueError: return None

def scrape_partite(driver, url, is_future_matches, year_context, codice_divisione_attuale):
    print(f"Tentativo di scraping da: {url}")
    try:
        driver.get(url)
    except Exception as e_get:
        print(f"Errore durante il caricamento dell'URL {url}: {e_get}")
        return [] 
        
    partite_data = []
    wait_time_after_load = random.uniform(2.5, 4.5) 
    time.sleep(wait_time_after_load) 

    try:
        WebDriverWait(driver, 25).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.event__match[id^='g_1_']")) 
        )
    except Exception as e:
        try:
            WebDriverWait(driver, 25).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[class*='event__match']")) 
            )
        except Exception as e2:
            print(f"Timeout o errore nell'attesa degli elementi partita su {url}: {e2}")
            return []

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    match_elements = soup.select("div.event__match[id^='g_1_']") 
    if not match_elements: 
        match_elements = soup.select("div[class*='event__match']")
    
    # print(f"Trovati {len(match_elements)} elementi 'match_elements' da analizzare su {url.split('/')[-2] if len(url.split('/')) > 2 else url}.") # Commentato

    for i, match_el in enumerate(match_elements): 
        current_date_str = None
        home_team = "N/A"
        away_team = "N/A"
        fthg, ftag, ftr = None, None, None
        
        try:
            date_in_row_el = match_el.select_one("div.event__time") 
            if date_in_row_el:
                date_raw_text = date_in_row_el.get_text(strip=True)
                parsed_dt_row = parse_date_flashscore(date_raw_text, year_context)
                if parsed_dt_row: current_date_str = parsed_dt_row
            if not current_date_str: continue 

            home_team_el_container = match_el.select_one("div.event__homeParticipant") 
            if home_team_el_container:
                name_el = home_team_el_container.select_one("span.wcl-name_3y6f5, strong.wcl-name_3y6f5, span[class*='wcl-name'], strong[class*='wcl-name'], span[data-testid='wcl-scores-simpleText-01'], span, strong")
                if name_el: home_team = name_el.get_text(strip=True)
                else: home_team = home_team_el_container.get_text(strip=True).strip() 
                if not home_team: home_team = "N/A"
            
            away_team_el_container = match_el.select_one("div.event__awayParticipant")
            if away_team_el_container:
                name_el = away_team_el_container.select_one("span.wcl-name_3y6f5, strong.wcl-name_3y6f5, span[class*='wcl-name'], strong[class*='wcl-name'], span[data-testid='wcl-scores-simpleText-01'], span, strong")
                if name_el: away_team = name_el.get_text(strip=True)
                else: away_team = away_team_el_container.get_text(strip=True).strip()
                if not away_team: away_team = "N/A"

            if home_team == "N/A" or away_team == "N/A": continue 
            
            if not is_future_matches:
                score_home_el = match_el.select_one("span.event__score--home")
                score_away_el = match_el.select_one("span.event__score--away")
                if score_home_el and score_away_el: 
                    fthg_text = score_home_el.get_text(strip=True)
                    ftag_text = score_away_el.get_text(strip=True)
                    if fthg_text.isdigit() and ftag_text.isdigit():
                        try:
                            fthg = int(fthg_text); ftag = int(ftag_text)
                            if fthg > ftag: ftr = 'H'
                            elif ftag > fthg: ftr = 'A'
                            else: ftr = 'D'
                        except ValueError: pass 
                    elif fthg_text or ftag_text : 
                        status_el = match_el.select_one("div[class*='event__stage']")
                        if status_el: 
                            status_text = status_el.get_text(strip=True).lower()
                            if any(kw in status_text for kw in ["post","ann","sosp","int","canc"]): ftr = "PST"
            
            partita = {
                'Div': codice_divisione_attuale, 'Date': current_date_str, 'HomeTeam': home_team, 
                'AwayTeam': away_team, 'FTHG': fthg, 'FTAG': ftag, 'FTR': ftr
            }
            if home_team != "N/A" and away_team != "N/A":
                partite_data.append(partita)
        except Exception as e_row:
            # print(f"Errore parsing riga {i+1} partita ({codice_divisione_attuale}): {e_row}") # Lasciato commentato
            # import traceback; traceback.print_exc() # Decommenta per debug errori riga specifici
            continue 
            
    print(f"Elaborazione {url.split('/')[-2] if len(url.split('/')) > 2 else url} completata. {len(partite_data)} partite aggiunte.")
    return partite_data

if __name__ == "__main__":
    start_time_script = time.time()
    driver = setup_driver() 
    
    if driver is None:
        print("Driver non inizializzato. Uscita."); exit() 
        
    all_matches_data_globale = [] 
    current_year = datetime.now().year 

    for campionato in CAMPIΟNATI_DA_SCRAPARE:
        print(f"\n--- INIZIO SCRAPING PER: {campionato['nome_visualizzato']} ({campionato['codice_divisione']}) ---")
        codice_div = campionato['codice_divisione']
        
        lista_dati_campionato_corrente = []
        urls_processati_per_campionato = set() # Per evitare di processare lo stesso URL più volte con la stessa logica

        # 1. URL Principale/Recente (se definito) - per risultati recenti e partite del giorno
        url_recente = campionato.get("url_principale_recente")
        if url_recente:
            print(f"  -> Processando URL principale/recente (per risultati recenti): {url_recente}")
            dati = scrape_partite(driver, url_recente, False, current_year, codice_div) # is_future_matches=False
            if dati: lista_dati_campionato_corrente.extend(dati)
            urls_processati_per_campionato.add((url_recente, False)) # Segna (url, is_future_matches) come processato
            time.sleep(random.uniform(1.5, 3.0))

        # 2. URL Storico Aggiuntivo (se definito e diverso da quello già processato per i risultati)
        url_storico = campionato.get("url_storico_aggiuntivo")
        if url_storico and (url_storico, False) not in urls_processati_per_campionato:
            print(f"  -> Processando URL storico aggiuntivo: {url_storico}")
            dati = scrape_partite(driver, url_storico, False, current_year, codice_div) # is_future_matches=False
            if dati: lista_dati_campionato_corrente.extend(dati)
            urls_processati_per_campionato.add((url_storico, False))
            time.sleep(random.uniform(1.5, 3.0))
        elif url_storico and (url_storico, False) in urls_processati_per_campionato:
            print(f"  -> URL storico aggiuntivo ({url_storico}) già processato per i risultati, skipping.")


        # 3. URL Calendario Esteso (se definito e diverso da quello già processato per il futuro)
        #    Oppure, se url_principale è definito, usalo anche per il futuro se url_calendario_esteso non c'è.
        url_futuro_da_usare = None
        url_calendario_cfg = campionato.get("url_calendario_esteso")
        
        if url_calendario_cfg:
            url_futuro_da_usare = url_calendario_cfg
            print_msg_futuro = f"  -> Processando URL calendario esteso: {url_futuro_da_usare}"
        elif url_recente: # Se non c'è un calendario esteso, prova ad usare l'url_principale anche per il futuro
            url_futuro_da_usare = url_recente
            print_msg_futuro = f"  -> Processando URL principale/recente (per calendario): {url_futuro_da_usare}"
        
        if url_futuro_da_usare and (url_futuro_da_usare, True) not in urls_processati_per_campionato:
            print(print_msg_futuro)
            dati = scrape_partite(driver, url_futuro_da_usare, True, current_year, codice_div) # is_future_matches=True
            if dati: lista_dati_campionato_corrente.extend(dati)
            urls_processati_per_campionato.add((url_futuro_da_usare, True))
        elif url_futuro_da_usare and (url_futuro_da_usare, True) in urls_processati_per_campionato:
            print(f"  -> URL ({url_futuro_da_usare}) già processato per il calendario, skipping.")
        elif not url_futuro_da_usare:
             print(f"  -> Nessun URL specificato o derivabile per il calendario futuro di {campionato['nome_visualizzato']}.")


        if lista_dati_campionato_corrente:
            all_matches_data_globale.extend(lista_dati_campionato_corrente)
            print(f"    Aggiunte {len(lista_dati_campionato_corrente)} righe (prima della deduplica globale) per {campionato['nome_visualizzato']}")
        
        print(f"--- FINE SCRAPING PER: {campionato['nome_visualizzato']} ---")
        time.sleep(random.uniform(2.0, 3.5)) 

    print("\nChiusura del driver del browser...")
    driver.quit() 

    if not all_matches_data_globale:
        print("\nNessun dato raccolto da nessun campionato.")
    else:
        df_globale = pd.DataFrame(all_matches_data_globale)
        print(f"\nDati grezzi totali raccolti: {len(df_globale)} righe.")

        # Strategia di deduplica migliorata:
        df_globale['FTR_is_null'] = df_globale['FTR'].isnull()
        df_globale.sort_values(by=['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTR_is_null'], 
                               ascending=[True, True, True, True, True],
                               inplace=True)
        df_globale.drop(columns=['FTR_is_null'], inplace=True)
        
        df_globale.drop_duplicates(subset=['Div', 'Date', 'HomeTeam', 'AwayTeam'], keep='first', inplace=True)
        print(f"Dati totali dopo rimozione duplicati: {len(df_globale)} righe.")
        
        if df_globale.empty:
            print("Il DataFrame globale è vuoto dopo la rimozione dei duplicati.")
        else:
            try:
                df_globale['SortDate'] = pd.to_datetime(df_globale['Date'], format='%d/%m/%y', errors='coerce')
                df_globale.dropna(subset=['SortDate'], inplace=True)
                if not df_globale.empty:
                    df_globale.sort_values(by=['Div', 'SortDate'], inplace=True) 
                    df_globale.drop(columns=['SortDate'], inplace=True)
            except Exception as e_sort_global:
                print(f"Errore sorting globale: {e_sort_global}")

            colonne_ordinate = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
            for col in colonne_ordinate:
                if col not in df_globale.columns: df_globale[col] = None 
            df_globale_final = df_globale[colonne_ordinate]
            
            try:
                df_globale_final.to_csv(OUTPUT_CSV_GLOBALE, index=False, encoding='utf-8')
                print(f"\nTutti i dati salvati con successo in {OUTPUT_CSV_GLOBALE} ({len(df_globale_final)} righe)")
                if not df_globale_final.empty:
                    print("\nPrime 5 righe del CSV globale:")
                    print(df_globale_final.head())
                    print("\nUltime 5 righe del CSV globale:")
                    print(df_globale_final.tail())
            except Exception as e_csv_global:
                print(f"Errore salvataggio CSV globale: {e_csv_global}")
    
    end_time_script = time.time()
    print(f"\nTempo totale di esecuzione dello script: {end_time_script - start_time_script:.2f} secondi.")