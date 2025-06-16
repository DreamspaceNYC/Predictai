import logging
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
HEADERS = {"User-Agent": USER_AGENT}


def _parse_match_string(match_str: str) -> Dict[str, str]:
    """Split a match string like 'Team A vs Team B' into components."""
    if " vs " in match_str:
        home, away = match_str.split(" vs ", 1)
        return {"match": match_str.strip(), "home_team": home.strip(), "away_team": away.strip()}
    return {"match": match_str.strip(), "home_team": match_str.strip(), "away_team": ""}


def scrape_betking_odds() -> List[Dict[str, float]]:
    """Scrape upcoming football odds from BetKing."""
    url = "https://www.betking.com/sportsbook/soccer"  # public landing page for matches
    results: List[Dict[str, float]] = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Example selectors based on typical bookmaker layout.
        for event in soup.select("div.event-holder"):
            match_name_el = event.select_one("div.event-name")
            odd_elements = event.select("span.odd-value")
            if not match_name_el or len(odd_elements) < 3:
                continue
            odds = []
            for el in odd_elements[:3]:
                try:
                    odds.append(float(el.get_text(strip=True)))
                except ValueError:
                    odds.append(None)
            match_info = _parse_match_string(match_name_el.get_text(strip=True))
            results.append({
                "match": match_info["match"],
                "home_odds": odds[0],
                "draw_odds": odds[1],
                "away_odds": odds[2],
            })
    except Exception as exc:
        logger.error("BetKing scraping failed: %s", exc)
    return results


def scrape_bet9ja_odds() -> List[Dict[str, float]]:
    """Scrape upcoming football odds from Bet9ja."""
    url = "https://web.bet9ja.com/Sport/Odds"  # generic sports odds page
    results: List[Dict[str, float]] = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for event in soup.select("div.event-item"):
            match_name_el = event.select_one("div.event-name")
            odd_elements = event.select("span.price")
            if not match_name_el or len(odd_elements) < 3:
                continue
            odds = []
            for el in odd_elements[:3]:
                try:
                    odds.append(float(el.get_text(strip=True)))
                except ValueError:
                    odds.append(None)
            match_info = _parse_match_string(match_name_el.get_text(strip=True))
            results.append({
                "match": match_info["match"],
                "home_odds": odds[0],
                "draw_odds": odds[1],
                "away_odds": odds[2],
            })
    except Exception as exc:
        logger.error("Bet9ja scraping failed: %s", exc)
    return results

