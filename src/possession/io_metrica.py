"""Loaders for Metrica Sports tracking and event data.

Two tracking formats are supported:

* the legacy CSV pair (one file per team) used by Metrica open sample games 1 and 2;
* FIFA/EPTS (metadata XML + raw text), used by sample game 3 **and by the hackathon ATD**.
  The EPTS reader is driven entirely by the metadata's DataFormatSpecification, so it handles
  ``x,y`` groups (sample data) and ``x,y,speed`` groups (hackathon ATD) without changes.

Every loader returns a :class:`possession.match.Match` indexed by ``frame_id - 1``.
"""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .match import Match

PITCH = (105.0, 68.0)  # Metrica sample data is normalised against a 105 x 68 m pitch
FPS = 25


# -- legacy CSV (sample games 1 and 2) -----------------------------------------------------------

def _read_team_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(frame_ids, period, players (N,P,2), ball (N,2)) from one team's tracking CSV."""
    table = pd.read_csv(path, skiprows=3, header=None, low_memory=False)
    period = table.iloc[:, 0].to_numpy(np.int8)
    frame_ids = table.iloc[:, 1].to_numpy(np.int64)
    coords = table.iloc[:, 3:].to_numpy(np.float32)
    n_players = coords.shape[1] // 2 - 1  # the last pair is the ball
    players = coords[:, : 2 * n_players].reshape(len(table), n_players, 2)
    ball = coords[:, 2 * n_players: 2 * n_players + 2]
    return frame_ids, period, players, ball


def load_csv_game(folder: Path, game: int) -> Match:
    folder = Path(folder)
    stem = f"Sample_Game_{game}"
    f_home, period, home, ball = _read_team_csv(folder / f"{stem}_RawTrackingData_Home_Team.csv")
    f_away, _, away, _ = _read_team_csv(folder / f"{stem}_RawTrackingData_Away_Team.csv")
    if not np.array_equal(f_home, f_away):
        raise ValueError("home and away tracking files are not frame-aligned")
    if not np.array_equal(f_home, np.arange(1, len(f_home) + 1)):
        raise ValueError("tracking frames are not contiguous from 1")

    players = np.concatenate([home, away], axis=1)
    team = np.array([0] * home.shape[1] + [1] * away.shape[1], dtype=np.int8)
    events = load_csv_events(folder / f"{stem}_RawEventsData.csv")
    return Match(name=f"metrica_g{game}", fps=FPS, pitch=PITCH, period=period, ball=ball,
                 players=players, team=team, events=events)


def load_csv_events(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return _normalise_events(pd.DataFrame({
        "team": raw["Team"].map({"Home": 0, "Away": 1}).fillna(-1).astype(int),
        "type": raw["Type"].astype(str),
        "subtype": raw["Subtype"].fillna("").astype(str),
        "period": raw["Period"].astype(int),
        "start_frame": raw["Start Frame"].astype(int),
        "end_frame": raw["End Frame"].astype(int),
    }))


# -- FIFA / EPTS (sample game 3, hackathon ATD) --------------------------------------------------

def read_epts_metadata(path: Path) -> dict:
    """Teams, players, channel layout per frame range, and period boundaries."""
    root = ET.parse(path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team_names = [t.findtext("Name") for t in root.iter("Team")]
    players = [(p.get("id"), p.get("teamId"), p.findtext("Name")) for p in root.iter("Player")]
    player_index = {pid: i for i, (pid, _, _) in enumerate(players)}
    channel = {c.get("id"): (c.get("playerId"), c.get("channelId"))
               for c in root.iter("PlayerChannel")}

    specs = []
    for spec in root.iter("DataFormatSpecification"):
        fields = []  # flat list of (kind, index, channel) in file order after the frame counter
        groups = list(spec)  # [StringRegister frameCount, SplitRegister players, SplitRegister ball]
        for node in groups[1:]:
            for ref in node.iter():
                if ref.tag == "PlayerChannelRef":
                    pid, ch = channel[ref.get("playerChannelId")]
                    fields.append(("player", player_index[pid], ch))
                elif ref.tag == "BallChannelRef":
                    fields.append(("ball", 0, ref.get("channelId")))
        specs.append({"start": int(spec.get("startFrame")), "end": int(spec.get("endFrame")),
                      "fields": fields})

    params = {p.findtext("Name"): p.findtext("Value")
              for g in root.iter("ProviderGlobalParameters") for p in g.iter("ProviderParameter")}
    periods = []
    for prefix in ("first_half", "second_half", "first_extra_half", "second_extra_half"):
        start, end = params.get(prefix + "_start"), params.get(prefix + "_end")
        if start and end and start.strip().lstrip("-").isdigit() and int(start) >= 0:
            periods.append((int(start), int(end)))

    fps = int(root.findtext(".//FrameRate") or FPS)
    return {"team_ids": team_ids, "team_names": team_names, "players": players,
            "specs": specs, "periods": periods, "fps": fps}


def load_epts(meta_path: Path, raw_path: Path, name: str, pitch=PITCH) -> Match:
    meta = read_epts_metadata(meta_path)
    home_id = meta["team_ids"][0]  # EPTS convention: first team listed is the home side
    team = np.array([0 if t == home_id else 1 for _, t, _ in meta["players"]], dtype=np.int8)

    text = Path(raw_path).read_text().replace(":", ",").replace(";", ",")
    width = max(len(s["fields"]) for s in meta["specs"]) + 1
    table = pd.read_csv(io.StringIO(text), header=None, names=range(width), na_values=["NaN"],
                        dtype=np.float64, engine="c")
    frame_ids = table[0].to_numpy(np.int64)
    n = int(frame_ids.max())
    values = table.to_numpy(np.float64)

    players = np.full((n, len(team), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    axis = {"x": 0, "y": 1}
    for spec in meta["specs"]:
        rows = np.flatnonzero((frame_ids >= spec["start"]) & (frame_ids <= spec["end"]))
        idx = frame_ids[rows] - 1
        for col, (kind, who, ch) in enumerate(spec["fields"], start=1):
            if ch not in axis:
                continue  # speed / z channels: speed is recomputed from positions downstream
            if kind == "player":
                players[idx, who, axis[ch]] = values[rows, col]
            else:
                ball[idx, axis[ch]] = values[rows, col]

    period = np.zeros(n, dtype=np.int8)
    for k, (start, end) in enumerate(meta["periods"], start=1):
        period[start - 1: min(end, n)] = k
    return Match(name=name, fps=meta["fps"], pitch=pitch, period=period, ball=ball,
                 players=players, team=team, meta={"epts": meta})


def load_json_events(path: Path, team_ids: list[str]) -> pd.DataFrame:
    """Metrica JSON events (sample game 3). Subtypes may be a dict, a list of dicts or null."""
    records = json.loads(Path(path).read_text())["data"]
    home_id = team_ids[0]
    rows = []
    for e in records:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = [sub]
        subtype = "-".join(s["name"] for s in sub) if sub else ""
        rows.append({
            "team": (0 if e["team"]["id"] == home_id else 1) if e.get("team") else -1,
            "type": e["type"]["name"],
            "subtype": subtype,
            "period": int(e["period"]),
            "start_frame": int(e["start"]["frame"]),
            "end_frame": int(e["end"]["frame"] or e["start"]["frame"]),
        })
    return _normalise_events(pd.DataFrame(rows))


def load_epts_game(folder: Path, game: int = 3) -> Match:
    folder = Path(folder)
    stem = f"Sample_Game_{game}"
    match = load_epts(folder / f"{stem}_metadata.xml", folder / f"{stem}_tracking.txt",
                      name=f"metrica_g{game}")
    match.events = load_json_events(folder / f"{stem}_events.json",
                                    match.meta["epts"]["team_ids"])
    return match


def _normalise_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["type"] = df["type"].str.upper().str.strip()
    df["subtype"] = df["subtype"].str.upper().str.strip()
    df["end_frame"] = np.maximum(df["end_frame"], df["start_frame"])
    return df.sort_values(["start_frame", "end_frame"], kind="stable").reset_index(drop=True)


# -- convenience -------------------------------------------------------------------------------

def load_sample_game(data_dir: Path, game: int) -> Match:
    data_dir = Path(data_dir)
    if game in (1, 2):
        return load_csv_game(data_dir / f"g{game}", game)
    if game == 3:
        return load_epts_game(data_dir / "g3", game)
    raise ValueError(f"unknown sample game {game}")
