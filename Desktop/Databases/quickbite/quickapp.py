from __future__ import annotations

import html
import sqlite3
from datetime import datetime, time
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from streamlit_folium import st_folium


st.set_page_config(
    page_title="MediTrack AI - Smart Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "meditrack.db"

BRAND = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "line": "rgba(148,163,184,0.22)",
    "card": "rgba(255,255,255,0.92)",
    "navy": "#071426",
    "teal": "#0f766e",
    "cyan": "#0891b2",
    "blue": "#2563eb",
    "green": "#16a34a",
    "violet": "#7c3aed",
    "amber": "#d97706",
    "rose": "#e11d48",
}

PAGE_COPY = {
    "Network Overview": {
        "eyebrow": "Executive Overview",
        "title": "MediTrack AI - Smart Healthcare Analytics",
        "hook": "A premium healthcare operations workspace for monitoring appointments, revenue, attendance reliability, and clinic performance across the network.",
        "story": "This page gives leadership a clean, strategic read on whether booking demand is translating into delivered care and monetized clinical throughput.",
        "takeaway": "Healthy growth comes from strong demand paired with clean attendance, disciplined clinic operations, and reliable schedule conversion.",
    },
    "Utilization & Revenue": {
        "eyebrow": "Operational Performance",
        "title": "Convert clinic demand into stronger utilization and revenue quality.",
        "hook": "This view pinpoints where schedules look busy on paper but underperform in completion, monetization, or wait-time discipline.",
        "story": "Not every full calendar is healthy. Utilization quality shows whether booked slots are actually turning into care delivered.",
        "takeaway": "The best operational wins usually come from fixing conversion in already-active clinics instead of chasing new demand from scratch.",
    },
    "No-Show Studio": {
        "eyebrow": "Predictive Intelligence",
        "title": "Predict no-show risk before schedule waste happens.",
        "hook": "MediTrack AI transforms historical appointment behavior into practical predictions that help teams intervene earlier and allocate capacity more intelligently.",
        "story": "Prediction is only valuable when it changes decisions. The point is to support better reminders, backup plans, and slot design before care is lost.",
        "takeaway": "High-risk bookings should trigger differentiated reminders, backup waitlists, and more deliberate scheduling decisions.",
    },
    "Doctor Explorer": {
        "eyebrow": "Care Team Lens",
        "title": "Understand which doctors create the strongest revenue and attendance outcomes.",
        "hook": "Doctor-level differences shape not just revenue, but patient confidence, show-up behavior, and the health of clinic capacity.",
        "story": "This view helps management benchmark clinicians, protect top performers, and coach where patient demand quality is softening.",
        "takeaway": "The best clinicians do more than generate revenue. They strengthen attendance quality and overall schedule resilience.",
    },
}

PAGE_IMAGES = {
    "Network Overview": [
        {
            "title": "Network command view",
            "caption": "A high-level view of appointments, revenue, and patient reliability across the clinic network.",
            "url": "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1200&q=80",
        },
        {
            "title": "Patient access",
            "caption": "Represents the patient journey from booking through attendance and care delivery.",
            "url": "https://images.unsplash.com/photo-1584515933487-779824d29309?auto=format&fit=crop&w=1200&q=80",
        },
        {
            "title": "Clinical analytics",
            "caption": "Signals data-driven operational decisions around throughput, staffing, and service quality.",
            "url": "https://images.unsplash.com/photo-1551076805-e1869033e561?auto=format&fit=crop&w=1200&q=80",
        },
    ],
    "Utilization & Revenue": [
        {
            "title": "Clinic throughput",
            "caption": "Shows how booked demand, wait times, and completions shape real clinic productivity.",
            "url": "https://images.unsplash.com/photo-1631815588090-d1bcbe9a0f5c?auto=format&fit=crop&w=1200&q=80",
        },
        {
            "title": "Doctor capacity",
            "caption": "Represents the care delivery capacity that leaders are trying to convert into revenue cleanly.",
            "url": "https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&w=1200&q=80",
        },
        {
            "title": "Revenue quality",
            "caption": "Reflects how fewer missed appointments improve commercial performance without adding new demand.",
            "url": "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?auto=format&fit=crop&w=1200&q=80",
        },
    ],
    "No-Show Studio": [
        {
            "title": "Predictive operations",
            "caption": "Highlights how historical booking behavior can support proactive scheduling intervention.",
            "url": "https://images.unsplash.com/photo-1579154204601-01588f351e67?auto=format&fit=crop&w=1200&q=80",
        },
        {
            "title": "Reminder workflows",
            "caption": "Represents the digital outreach and confirmation steps triggered by no-show risk.",
            "url": "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=1200&q=80",
        },
        {
            "title": "AI-assisted planning",
            "caption": "Shows the planning mindset behind smarter slot allocation, waitlists, and staffing.",
            "url": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1200&q=80",
        },
    ],
    "Doctor Explorer": [
        {
            "title": "Clinical excellence",
            "caption": "Represents doctors who combine strong attendance reliability with patient trust.",
            "url": "https://images.unsplash.com/photo-1612277795421-9bc7706a4a41?auto=format&fit=crop&w=1200&q=80",
        },
        {
            "title": "Patient confidence",
            "caption": "Highlights how a strong patient experience reinforces attendance behavior and retention.",
            "url": "https://images.unsplash.com/photo-1579684385127-1ef15d508118?auto=format&fit=crop&w=1200&q=80",
        },
        {
            "title": "Benchmarking view",
            "caption": "Captures the performance comparison lens used to evaluate clinicians across departments.",
            "url": "https://images.unsplash.com/photo-1518186285589-2f7649de83e0?auto=format&fit=crop&w=1200&q=80",
        },
    ],
}


def inject_styles() -> None:
    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Manrope:wght@400;500;600;700&display=swap');

        :root {{
            --ink: {BRAND["ink"]};
            --muted: {BRAND["muted"]};
            --line: {BRAND["line"]};
            --card: {BRAND["card"]};
            --navy: {BRAND["navy"]};
            --teal: {BRAND["teal"]};
            --cyan: {BRAND["cyan"]};
            --blue: {BRAND["blue"]};
            --green: {BRAND["green"]};
            --amber: {BRAND["amber"]};
            --rose: {BRAND["rose"]};
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(8,145,178,0.13), transparent 24%),
                radial-gradient(circle at top right, rgba(37,99,235,0.12), transparent 22%),
                linear-gradient(180deg, #fbfdff 0%, #f4f8fc 48%, #eef5fb 100%);
            color: var(--ink);
        }}

        .block-container {{
            max-width: 1240px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }}

        h1, h2, h3, h4 {{
            font-family: "Sora", sans-serif;
            color: var(--ink);
            letter-spacing: -0.04em;
        }}

        p, li, label, span, div {{
            font-family: "Manrope", sans-serif;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #06101f 0%, #0c1b31 56%, #123153 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: #f8fafc !important;
        }}

        .top-brand {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
            padding: 1.15rem 1.2rem;
            border-radius: 28px;
            background:
                radial-gradient(circle at top right, rgba(124,58,237,0.16), transparent 22%),
                linear-gradient(135deg, rgba(255,255,255,0.94), rgba(235,248,255,0.90));
            border: 1px solid rgba(37,99,235,0.10);
            backdrop-filter: blur(12px);
            box-shadow: 0 24px 50px rgba(15,23,42,0.08);
        }}

        .brand-shell {{
            display: flex;
            gap: 0.9rem;
            align-items: center;
            margin-bottom: 1rem;
        }}

        .brand-mark,
        .hero-mark {{
            width: 58px;
            height: 58px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(221,244,255,0.95));
            box-shadow: 0 18px 42px rgba(0,0,0,0.18);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--blue);
            font-family: "Sora", sans-serif;
            font-weight: 800;
            font-size: 1.2rem;
            flex-shrink: 0;
        }}

        .hero-mark {{
            width: 64px;
            height: 64px;
            color: var(--teal);
        }}

        .brand-name {{
            font-family: "Sora", sans-serif;
            font-size: 1.08rem;
            font-weight: 700;
            color: white;
            margin: 0;
        }}

        .brand-copy {{
            margin: 0.1rem 0 0 0;
            color: rgba(255,255,255,0.72);
            font-size: 0.9rem;
        }}

        .main-brand-title {{
            margin: 0;
            font-family: "Sora", sans-serif;
            font-size: 1.28rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--blue), var(--teal), var(--violet));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .main-brand-copy {{
            margin: 0.12rem 0 0 0;
            color: var(--muted);
            line-height: 1.7;
            font-size: 0.95rem;
        }}

        .top-pill {{
            padding: 0.55rem 0.85rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(15,118,110,0.10), rgba(37,99,235,0.08));
            border: 1px solid rgba(15,118,110,0.12);
            color: var(--teal);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            white-space: nowrap;
        }}

        .sidebar-card {{
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 1rem;
            margin: 0.75rem 0 1rem 0;
        }}

        .sidebar-title {{
            color: rgba(255,255,255,0.68);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            margin-bottom: 0.35rem;
        }}

        .sidebar-value {{
            color: white;
            font-family: "Sora", sans-serif;
            font-size: 1.3rem;
            margin-bottom: 0.25rem;
        }}

        .hero {{
            position: relative;
            overflow: hidden;
            border-radius: 30px;
            padding: 2.3rem;
            margin-bottom: 1.2rem;
            background:
                radial-gradient(circle at 85% 20%, rgba(255,255,255,0.22), transparent 18%),
                linear-gradient(135deg, rgba(6,16,31,0.99) 0%, rgba(11,28,49,0.97) 48%, rgba(8,145,178,0.90) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 26px 70px rgba(15,23,42,0.16);
        }}

        .hero-grid {{
            display: grid;
            grid-template-columns: 1.08fr 0.92fr;
            gap: 1.2rem;
            align-items: stretch;
        }}

        .eyebrow {{
            display: inline-block;
            color: rgba(255,255,255,0.78);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.72rem;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }}

        .hero-brand-row {{
            display: flex;
            align-items: center;
            gap: 0.9rem;
            margin-bottom: 1rem;
        }}

        .hero-title {{
            font-size: clamp(2.5rem, 4vw, 4.1rem);
            line-height: 1.02;
            color: white;
            margin: 0 0 0.8rem 0;
            max-width: 780px;
        }}

        .hero-copy {{
            color: rgba(255,255,255,0.82);
            max-width: 680px;
            line-height: 1.78;
            font-size: 1.02rem;
            margin: 0 0 1rem 0;
        }}

        .hero-mini-stats {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }}

        .mini-stat {{
            padding: 0.9rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
            backdrop-filter: blur(8px);
        }}

        .mini-label {{
            color: rgba(255,255,255,0.66);
            font-size: 0.78rem;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .mini-value {{
            color: white;
            font-family: "Sora", sans-serif;
            font-size: 1.28rem;
        }}

        .hero-banner-card {{
            position: relative;
            min-height: 360px;
            border-radius: 26px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 22px 46px rgba(15,23,42,0.12);
        }}

        .hero-banner-card img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}

        .hero-banner-card::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(7,20,38,0.06) 0%, rgba(7,20,38,0.80) 100%);
        }}

        .hero-banner-caption {{
            position: absolute;
            left: 1.2rem;
            right: 1.2rem;
            bottom: 1.2rem;
            z-index: 2;
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.12);
            backdrop-filter: blur(10px);
            color: white;
        }}

        .section-label {{
            display: block;
            margin: 0.3rem 0 0.95rem 0;
            font-family: "Sora", sans-serif;
            font-weight: 800;
            font-size: 1.35rem;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, var(--blue), var(--teal), var(--violet));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .section-rule {{
            height: 1px;
            width: 100%;
            margin: 0.1rem 0 1.1rem 0;
            background: linear-gradient(90deg, rgba(37,99,235,0.65), rgba(8,145,178,0.32), transparent 88%);
        }}

        .story-card {{
            background: linear-gradient(135deg, rgba(255,255,255,0.97), rgba(240,249,255,0.92));
            border: 1px solid rgba(8,145,178,0.14);
            border-radius: 26px;
            padding: 1.35rem 1.5rem;
            box-shadow: 0 18px 40px rgba(15,23,42,0.05);
            margin: 1rem 0 1.1rem 0;
        }}

        .story-title {{
            font-family: "Sora", sans-serif;
            font-size: 1.12rem;
            margin-bottom: 0.3rem;
        }}

        .story-copy {{
            margin: 0;
            color: var(--muted);
            line-height: 1.8;
        }}

        .metric-card {{
            background:
                radial-gradient(circle at top right, rgba(124,58,237,0.11), transparent 24%),
                linear-gradient(135deg, rgba(255,255,255,0.98), rgba(239,246,255,0.94));
            border: 1px solid rgba(37,99,235,0.10);
            border-radius: 24px;
            padding: 1.2rem;
            min-height: 154px;
            box-shadow: 0 20px 44px rgba(15,23,42,0.08);
        }}

        .metric-label {{
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            font-size: 0.76rem;
            margin-bottom: 0.5rem;
        }}

        .metric-value {{
            font-family: "Sora", sans-serif;
            font-size: 2rem;
            color: var(--ink);
            margin-bottom: 0.25rem;
        }}

        .metric-delta {{
            color: var(--teal);
            line-height: 1.7;
            font-size: 0.94rem;
        }}

        .insight-card {{
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.15rem;
            height: 100%;
            box-shadow: 0 14px 32px rgba(15,23,42,0.05);
        }}

        .insight-tag {{
            color: var(--cyan);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-size: 0.72rem;
            margin-bottom: 0.35rem;
        }}

        .insight-title {{
            font-family: "Sora", sans-serif;
            font-size: 1.1rem;
            margin-bottom: 0.3rem;
            color: var(--ink);
        }}

        .insight-body {{
            color: var(--muted);
            line-height: 1.72;
            font-size: 0.94rem;
            margin: 0;
        }}

        .summary-card {{
            background: rgba(255,255,255,0.95);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.2rem;
            box-shadow: 0 16px 32px rgba(15,23,42,0.05);
        }}

        .summary-card ul {{
            padding-left: 1rem;
            margin-bottom: 0;
        }}

        .settings-panel {{
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 1rem;
            margin: 0.85rem 0 1rem 0;
        }}

        .settings-title {{
            color: rgba(255,255,255,0.68);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            margin-bottom: 0.65rem;
        }}

        .settings-copy {{
            color: rgba(255,255,255,0.72);
            line-height: 1.65;
            font-size: 0.88rem;
            margin: 0.25rem 0 0.8rem 0;
        }}

        .image-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin: 0.8rem 0 1.15rem 0;
        }}

        .image-card {{
            position: relative;
            min-height: 260px;
            border-radius: 24px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.18);
            box-shadow: 0 22px 46px rgba(15,23,42,0.09);
            transform: translateY(0);
            transition: transform 180ms ease, box-shadow 180ms ease;
            background: #dbeafe;
            isolation: isolate;
        }}

        .image-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 28px 58px rgba(15,23,42,0.13);
        }}

        .image-card img {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}

        .image-card::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(7,20,38,0.04) 20%, rgba(7,20,38,0.72) 100%);
            pointer-events: none;
        }}

        .hover-arrow {{
            position: absolute;
            right: 0.9rem;
            top: 0.9rem;
            z-index: 3;
            padding: 0.42rem 0.65rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.92);
            color: var(--ink);
            font-size: 0.75rem;
            font-weight: 700;
            box-shadow: 0 10px 20px rgba(15,23,42,0.16);
        }}

        .hover-tooltip {{
            position: absolute;
            left: 1rem;
            right: 1rem;
            bottom: 1rem;
            z-index: 3;
            padding: 1rem;
            border-radius: 18px;
            background: rgba(7,20,38,0.82);
            color: white;
            border: 1px solid rgba(255,255,255,0.12);
            backdrop-filter: blur(14px);
            opacity: 0;
            transform: translateY(14px);
            transition: opacity 200ms ease, transform 200ms ease;
            pointer-events: none;
        }}

        .image-card:hover .hover-tooltip {{
            opacity: 1;
            transform: translateY(0);
        }}

        .hover-title {{
            font-family: "Sora", sans-serif;
            font-size: 1rem;
            margin-bottom: 0.22rem;
        }}

        .hover-copy {{
            margin: 0;
            color: rgba(255,255,255,0.84);
            font-size: 0.9rem;
            line-height: 1.58;
        }}

        .takeaway {{
            background: linear-gradient(135deg, rgba(22,163,74,0.08), rgba(8,145,178,0.08));
            border: 1px solid rgba(15,118,110,0.14);
            border-radius: 26px;
            padding: 1.35rem 1.45rem;
            margin-top: 1rem;
        }}

        .takeaway-title {{
            font-family: "Sora", sans-serif;
            margin-bottom: 0.3rem;
        }}

        .takeaway-copy {{
            margin: 0;
            color: var(--muted);
            line-height: 1.78;
        }}

        .chart-note {{
            color: var(--muted);
            line-height: 1.72;
            margin-top: -0.15rem;
            margin-bottom: 0.8rem;
        }}

        .form-shell {{
            background: rgba(255,255,255,0.94);
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 16px 34px rgba(15,23,42,0.04);
        }}

        .assistant-shell {{
            background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.05));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 1rem;
            margin-top: 1rem;
        }}

        .assistant-title {{
            color: white;
            font-family: "Sora", sans-serif;
            font-size: 1rem;
            margin-bottom: 0.25rem;
        }}

        .assistant-copy {{
            color: rgba(255,255,255,0.72);
            line-height: 1.65;
            font-size: 0.88rem;
            margin-bottom: 0.85rem;
        }}

        .floating-chat-panel {{
            position: fixed;
            right: 20px;
            bottom: 96px;
            width: 320px;
            height: 78vh;
            max-height: 780px;
            z-index: 9997;
            border-radius: 24px;
            background: rgba(255,255,255,0.98);
            border: 1px solid rgba(148,163,184,0.18);
            box-shadow: 0 26px 70px rgba(15,23,42,0.18);
            overflow: hidden;
            backdrop-filter: blur(14px);
        }}

        .floating-chat-header {{
            padding: 1rem 1rem 0.85rem 1rem;
            background: linear-gradient(135deg, rgba(37,99,235,0.96), rgba(8,145,178,0.96));
            color: white;
            font-family: "Sora", sans-serif;
            font-size: 1rem;
            font-weight: 700;
        }}

        .floating-chat-subtitle {{
            font-family: "Manrope", sans-serif;
            font-size: 0.8rem;
            color: rgba(255,255,255,0.76);
            margin-top: 0.22rem;
        }}

        .floating-chat-body {{
            padding: 0.95rem 1rem 6.8rem 1rem;
            height: calc(78vh - 84px);
            overflow-y: auto;
            background:
                url('https://images.unsplash.com/photo-1485827404703-89b55fcc595e?auto=format&fit=crop&w=400&q=80'),
                radial-gradient(circle at top right, rgba(124,58,237,0.08), transparent 22%),
                linear-gradient(180deg, #f8fbff 0%, #f3f8fd 100%);
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .chat-row {{
            display: flex;
            margin-bottom: 0.7rem;
        }}

        .chat-row.user {{
            justify-content: flex-end;
        }}

        .chat-row.bot {{
            justify-content: flex-start;
        }}

        .chat-bubble {{
            max-width: 86%;
            padding: 0.72rem 0.85rem;
            border-radius: 18px;
            line-height: 1.6;
            font-size: 0.9rem;
            box-shadow: 0 10px 20px rgba(15,23,42,0.08);
        }}

        .chat-row.user .chat-bubble {{
            background: linear-gradient(135deg, rgba(37,99,235,0.98), rgba(8,145,178,0.92));
            color: white;
            border-bottom-right-radius: 8px;
        }}

        .chat-row.bot .chat-bubble {{
            background: white;
            color: var(--ink);
            border: 1px solid rgba(148,163,184,0.16);
            border-bottom-left-radius: 8px;
        }}

        .st-key-floating_chat_open {{
            position: fixed;
            right: 20px;
            bottom: 20px;
            z-index: 9998;
            width: 68px;
        }}

        .st-key-floating_chat_open button {{
            width: 68px;
            height: 68px;
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, var(--blue), var(--teal), var(--violet));
            color: white;
            font-size: 1.65rem;
            box-shadow: 0 18px 34px rgba(37,99,235,0.35);
        }}

        .st-key-floating_chat_close {{
            position: fixed;
            right: 36px;
            bottom: calc(78vh + 24px);
            z-index: 9999;
            width: 44px;
        }}

        .st-key-floating_chat_close button {{
            width: 42px;
            height: 42px;
            border-radius: 999px;
            border: none;
            background: rgba(255,255,255,0.18);
            color: white;
            font-size: 1rem;
        }}

        .st-key-floating_chat_input {{
            position: fixed;
            right: 34px;
            bottom: 86px;
            width: 228px;
            z-index: 9999;
        }}

        .st-key-floating_chat_input input {{
            background: white;
            border-radius: 14px;
            border: 1px solid rgba(148,163,184,0.24);
            color: black;
        }}

        .st-key-floating_chat_send {{
            position: fixed;
            right: 34px;
            bottom: 34px;
            width: 252px;
            z-index: 9999;
        }}

        .st-key-floating_chat_send button {{
            width: 100%;
            border-radius: 14px;
            border: none;
            background: linear-gradient(135deg, var(--blue), var(--teal));
            color: white;
            font-weight: 700;
        }}

        .auto-insight {{
            padding: 0.95rem 1rem;
            border-radius: 18px;
            margin: 0.85rem 0 1rem 0;
            border: 1px solid rgba(37,99,235,0.10);
            background:
                radial-gradient(circle at right top, rgba(124,58,237,0.10), transparent 24%),
                linear-gradient(135deg, rgba(255,255,255,0.97), rgba(240,249,255,0.94));
            box-shadow: 0 16px 30px rgba(15,23,42,0.06);
        }}

        .auto-insight-title {{
            font-family: "Sora", sans-serif;
            font-size: 0.95rem;
            color: var(--ink);
            margin-bottom: 0.2rem;
        }}

        .auto-insight-copy {{
            margin: 0;
            color: var(--muted);
            line-height: 1.65;
        }}

        .warning-board,
        .recommendation-board {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin: 0.85rem 0 1.15rem 0;
        }}

        .warning-card,
        .recommendation-card {{
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 16px 34px rgba(15,23,42,0.05);
        }}

        .warning-card {{
            background: linear-gradient(135deg, rgba(255,251,235,0.96), rgba(255,255,255,0.95));
            border: 1px solid rgba(217,119,6,0.20);
        }}

        .warning-card.is-critical {{
            background: linear-gradient(135deg, rgba(255,241,242,0.98), rgba(255,255,255,0.95));
            border-color: rgba(225,29,72,0.22);
        }}

        .recommendation-card {{
            background: linear-gradient(135deg, rgba(236,253,245,0.96), rgba(240,249,255,0.96));
            border: 1px solid rgba(15,118,110,0.18);
        }}

        .signal-tag {{
            display: inline-block;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }}

        .warning-card .signal-tag {{
            color: var(--amber);
            background: rgba(245,158,11,0.10);
        }}

        .warning-card.is-critical .signal-tag {{
            color: var(--rose);
            background: rgba(225,29,72,0.10);
        }}

        .recommendation-card .signal-tag {{
            color: var(--teal);
            background: rgba(15,118,110,0.10);
        }}

        .signal-title {{
            font-family: "Sora", sans-serif;
            font-size: 1.05rem;
            color: var(--ink);
            margin-bottom: 0.28rem;
        }}

        .signal-body {{
            margin: 0;
            color: var(--muted);
            line-height: 1.72;
        }}

        .footer {{
            margin-top: 2.2rem;
            padding: 2rem 1.2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(7,20,38,0.98), rgba(11,28,49,0.96));
            border: 1px solid rgba(255,255,255,0.06);
            text-align: center;
            color: rgba(255,255,255,0.80);
            box-shadow: 0 20px 44px rgba(15,23,42,0.10);
        }}

        .footer h3 {{
            color: white;
            margin-bottom: 0.25rem;
        }}

        .footer a {{
            color: #7dd3fc;
            text-decoration: none;
            margin: 0 0.45rem;
            font-weight: 600;
        }}

        .footer p {{
            margin: 0.28rem 0;
            color: rgba(255,255,255,0.76);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.75rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 999px;
            padding: 0.6rem 0.95rem;
            font-weight: 700;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(15,118,110,0.10), rgba(37,99,235,0.12));
            border-color: rgba(15,118,110,0.16);
            color: var(--teal);
        }}

        @media (max-width: 980px) {{
            .hero-grid {{
                grid-template-columns: 1fr;
            }}

            .image-grid {{
                grid-template-columns: 1fr;
            }}

            .hero-mini-stats {{
                grid-template-columns: 1fr;
            }}

            .top-brand {{
                flex-direction: column;
                align-items: flex-start;
            }}

            .floating-chat-panel {{
                width: calc(100vw - 24px);
                right: 12px;
                left: 12px;
                bottom: 88px;
                height: 72vh;
            }}

            .st-key-floating_chat_input {{
                right: 26px;
                left: 26px;
                width: auto;
            }}

            .st-key-floating_chat_send {{
                right: 26px;
                left: 26px;
                width: auto;
            }}
        }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_database() -> bool:
    return DB_PATH.exists()


def ensure_appointments_support_scheduled() -> None:
    with get_connection() as conn:
        create_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='appointments'"
        ).fetchone()
        if not create_sql:
            return
        if "Scheduled" in create_sql[0]:
            return

        conn.executescript(
            """
            BEGIN TRANSACTION;
            CREATE TABLE appointments_new (
                appointment_id INTEGER PRIMARY KEY,
                appointment_booked_at TEXT NOT NULL,
                appointment_scheduled_at TEXT NOT NULL,
                appointment_date TEXT NOT NULL,
                patient_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                clinic_id INTEGER NOT NULL,
                department_id INTEGER NOT NULL,
                city_id INTEGER NOT NULL,
                appointment_channel TEXT NOT NULL,
                patient_type TEXT NOT NULL CHECK (patient_type IN ('New', 'Returning')),
                appointment_hour INTEGER NOT NULL,
                day_of_week TEXT NOT NULL,
                month_name TEXT NOT NULL,
                lead_days INTEGER NOT NULL,
                consultation_fee REAL NOT NULL,
                appointment_status TEXT NOT NULL CHECK (appointment_status IN ('Scheduled', 'Completed', 'Cancelled', 'No-show')),
                wait_time_minutes REAL NOT NULL,
                consultation_duration_minutes REAL NOT NULL,
                satisfaction_score REAL,
                no_show_risk_score REAL NOT NULL,
                is_peak_slot INTEGER NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id),
                FOREIGN KEY (clinic_id) REFERENCES clinics(clinic_id),
                FOREIGN KEY (department_id) REFERENCES departments(department_id),
                FOREIGN KEY (city_id) REFERENCES cities(city_id)
            );
            INSERT INTO appointments_new
            SELECT * FROM appointments;
            DROP TABLE appointments;
            ALTER TABLE appointments_new RENAME TO appointments;
            COMMIT;
            """
        )


def chart_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        margin=dict(l=10, r=10, t=58, b=10),
        font=dict(color=BRAND["ink"], family="Manrope"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="white", font_color=BRAND["ink"]),
        height=560,
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def build_where_clause(city: str, department: str, start_date, end_date) -> tuple[str, list]:
    clauses = []
    params: list = []
    if city != "All":
        clauses.append("c.city_name = ?")
        params.append(city)
    if department != "All":
        clauses.append("d.department_name = ?")
        params.append(department)
    clauses.append("DATE(a.appointment_date) BETWEEN DATE(?) AND DATE(?)")
    params.extend([str(start_date), str(end_date)])

    where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
    return where_sql, params


def analytics_base_sql(where_sql: str = "") -> str:
    return f"""
        FROM appointments a
        JOIN cities c ON a.city_id = c.city_id
        JOIN clinics cl ON a.clinic_id = cl.clinic_id
        JOIN departments d ON a.department_id = d.department_id
        JOIN doctors doc ON a.doctor_id = doc.doctor_id
        JOIN patients p ON a.patient_id = p.patient_id
        {where_sql}
    """


@st.cache_data(show_spinner=False)
def get_filter_options():
    with get_connection() as conn:
        cities = pd.read_sql_query("SELECT city_name FROM cities ORDER BY city_name", conn)["city_name"].tolist()
        departments = pd.read_sql_query(
            "SELECT department_name FROM departments ORDER BY department_name", conn
        )["department_name"].tolist()
        date_bounds = pd.read_sql_query(
            "SELECT MIN(DATE(appointment_date)) AS min_date, MAX(DATE(appointment_date)) AS max_date FROM appointments",
            conn,
        )
    return cities, departments, date_bounds.iloc[0]["min_date"], date_bounds.iloc[0]["max_date"]


@st.cache_data(show_spinner=False)
def load_detail_data(city: str, department: str, start_date: str, end_date: str) -> pd.DataFrame:
    where_sql, params = build_where_clause(city, department, start_date, end_date)
    query = f"""
        SELECT
            a.appointment_id,
            a.appointment_booked_at,
            a.appointment_scheduled_at,
            a.appointment_date,
            a.appointment_channel,
            a.patient_type,
            a.appointment_hour,
            a.day_of_week,
            a.month_name,
            a.lead_days,
            a.consultation_fee,
            a.appointment_status,
            a.wait_time_minutes,
            a.consultation_duration_minutes,
            a.satisfaction_score,
            a.no_show_risk_score,
            a.is_peak_slot,
            c.city_name,
            c.province,
            c.traffic_index,
            cl.clinic_name,
            cl.rooms_count,
            d.department_name,
            d.base_fee,
            doc.doctor_name,
            doc.seniority_level,
            doc.years_experience,
            doc.popularity_score,
            p.patient_segment,
            p.gender AS patient_gender,
            p.chronic_flag
        {analytics_base_sql(where_sql)}
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    df["appointment_booked_at"] = pd.to_datetime(df["appointment_booked_at"])
    df["appointment_scheduled_at"] = pd.to_datetime(df["appointment_scheduled_at"])
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    df["month"] = df["appointment_date"].dt.to_period("M").astype(str)
    df["is_completed"] = (df["appointment_status"] == "Completed").astype(int)
    df["is_cancelled"] = (df["appointment_status"] == "Cancelled").astype(int)
    df["is_no_show"] = (df["appointment_status"] == "No-show").astype(int)
    df["is_scheduled"] = (df["appointment_status"] == "Scheduled").astype(int)
    df["realized_revenue"] = np.where(df["appointment_status"] == "Completed", df["consultation_fee"], 0.0)
    return df


@st.cache_data(show_spinner=False)
def query_snapshot(city: str, department: str, start_date: str, end_date: str) -> pd.DataFrame:
    where_sql, params = build_where_clause(city, department, start_date, end_date)
    query = f"""
        SELECT
            COUNT(*) AS appointments,
            SUM(CASE WHEN a.appointment_status = 'Completed' THEN a.consultation_fee ELSE 0 END) AS realized_revenue,
            AVG(CASE WHEN a.appointment_status = 'No-show' THEN 1.0 ELSE 0.0 END) AS no_show_rate,
            AVG(CASE WHEN a.appointment_status = 'Completed' THEN 1.0 ELSE 0.0 END) AS completion_rate,
            AVG(CASE WHEN a.appointment_status = 'Completed' THEN a.wait_time_minutes END) AS avg_wait_time
        {analytics_base_sql(where_sql)}
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


@st.cache_data(show_spinner=False)
def query_monthly_metrics(city: str, department: str, start_date: str, end_date: str) -> pd.DataFrame:
    where_sql, params = build_where_clause(city, department, start_date, end_date)
    query = f"""
        SELECT
            strftime('%Y-%m', a.appointment_date) AS month,
            COUNT(*) AS appointments,
            SUM(CASE WHEN a.appointment_status = 'Completed' THEN 1 ELSE 0 END) AS completed,
            SUM(CASE WHEN a.appointment_status = 'Completed' THEN a.consultation_fee ELSE 0 END) AS revenue,
            AVG(CASE WHEN a.appointment_status = 'No-show' THEN 1.0 ELSE 0.0 END) AS no_show_rate
        {analytics_base_sql(where_sql)}
        GROUP BY strftime('%Y-%m', a.appointment_date)
        ORDER BY month
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


@st.cache_data(show_spinner=False)
def query_city_metrics(city: str, department: str, start_date: str, end_date: str) -> pd.DataFrame:
    where_sql, params = build_where_clause(city, department, start_date, end_date)
    query = f"""
        SELECT
            c.city_name,
            COUNT(*) AS appointments,
            SUM(CASE WHEN a.appointment_status = 'Completed' THEN 1 ELSE 0 END) AS completed,
            SUM(CASE WHEN a.appointment_status = 'Completed' THEN a.consultation_fee ELSE 0 END) AS revenue,
            AVG(CASE WHEN a.appointment_status = 'No-show' THEN 1.0 ELSE 0.0 END) AS no_show_rate,
            AVG(CASE WHEN a.appointment_status = 'Completed' THEN a.wait_time_minutes END) AS avg_wait
        {analytics_base_sql(where_sql)}
        GROUP BY c.city_name
        ORDER BY revenue DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


@st.cache_data(show_spinner=False)
def query_department_metrics(city: str, department: str, start_date: str, end_date: str) -> pd.DataFrame:
    where_sql, params = build_where_clause(city, department, start_date, end_date)
    query = f"""
        SELECT
            d.department_name,
            COUNT(*) AS appointments,
            SUM(CASE WHEN a.appointment_status = 'Completed' THEN 1 ELSE 0 END) AS completed,
            AVG(CASE WHEN a.appointment_status = 'No-show' THEN 1.0 ELSE 0.0 END) AS no_show_rate,
            AVG(a.consultation_fee) AS avg_fee,
            SUM(CASE WHEN a.appointment_status = 'Completed' THEN a.consultation_fee ELSE 0 END) AS revenue
        {analytics_base_sql(where_sql)}
        GROUP BY d.department_name
        ORDER BY revenue DESC
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    df["utilization_rate"] = np.where(df["appointments"] > 0, df["completed"] / df["appointments"], 0.0)
    return df


@st.cache_data(show_spinner=False)
def query_clinic_metrics(city: str, department: str, start_date: str, end_date: str) -> pd.DataFrame:
    where_sql, params = build_where_clause(city, department, start_date, end_date)
    query = f"""
        SELECT
            c.city_name,
            cl.clinic_name,
            COUNT(*) AS appointments,
            SUM(CASE WHEN a.appointment_status = 'Completed' THEN 1 ELSE 0 END) AS completed,
            SUM(CASE WHEN a.appointment_status = 'Completed' THEN a.consultation_fee ELSE 0 END) AS revenue,
            AVG(CASE WHEN a.appointment_status = 'No-show' THEN 1.0 ELSE 0.0 END) AS no_show_rate,
            AVG(CASE WHEN a.appointment_status = 'Completed' THEN a.wait_time_minutes END) AS avg_wait
        {analytics_base_sql(where_sql)}
        GROUP BY c.city_name, cl.clinic_name
        ORDER BY revenue DESC
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    df["completion_rate"] = np.where(df["appointments"] > 0, df["completed"] / df["appointments"], 0.0)
    return df


@st.cache_data(show_spinner=False)
def query_no_show_by_type(city: str, department: str, start_date: str, end_date: str) -> pd.DataFrame:
    where_sql, params = build_where_clause(city, department, start_date, end_date)
    query = f"""
        SELECT
            a.patient_type,
            d.department_name,
            AVG(CASE WHEN a.appointment_status = 'No-show' THEN 1.0 ELSE 0.0 END) AS no_show_rate,
            COUNT(*) AS appointments
        {analytics_base_sql(where_sql)}
        GROUP BY a.patient_type, d.department_name
        ORDER BY no_show_rate DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


@st.cache_data(show_spinner=False)
def query_no_show_by_hour(city: str, department: str, start_date: str, end_date: str) -> pd.DataFrame:
    where_sql, params = build_where_clause(city, department, start_date, end_date)
    query = f"""
        SELECT
            a.appointment_hour,
            AVG(CASE WHEN a.appointment_status = 'No-show' THEN 1.0 ELSE 0.0 END) AS no_show_rate,
            COUNT(*) AS appointments
        {analytics_base_sql(where_sql)}
        GROUP BY a.appointment_hour
        ORDER BY a.appointment_hour
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


@st.cache_data(show_spinner=False)
def query_doctor_metrics(city: str, department: str, start_date: str, end_date: str) -> pd.DataFrame:
    where_sql, params = build_where_clause(city, department, start_date, end_date)
    query = f"""
        SELECT
            doc.doctor_name,
            d.department_name,
            c.city_name,
            doc.seniority_level,
            COUNT(*) AS appointments,
            SUM(CASE WHEN a.appointment_status = 'Completed' THEN 1 ELSE 0 END) AS completed,
            AVG(CASE WHEN a.appointment_status = 'No-show' THEN 1.0 ELSE 0.0 END) AS no_show_rate,
            SUM(CASE WHEN a.appointment_status = 'Completed' THEN a.consultation_fee ELSE 0 END) AS revenue,
            AVG(a.satisfaction_score) AS avg_satisfaction,
            AVG(CASE WHEN a.appointment_status = 'Completed' THEN a.wait_time_minutes END) AS avg_wait
        {analytics_base_sql(where_sql)}
        GROUP BY doc.doctor_name, d.department_name, c.city_name, doc.seniority_level
        ORDER BY revenue DESC
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    df["completion_rate"] = np.where(df["appointments"] > 0, df["completed"] / df["appointments"], 0.0)
    return df


@st.cache_data(show_spinner=False)
def query_doctor_monthly(doctor_name: str, city: str, department: str, start_date: str, end_date: str) -> pd.DataFrame:
    where_sql, params = build_where_clause(city, department, start_date, end_date)
    where_sql = f"{where_sql} AND doc.doctor_name = ?"
    params = params + [doctor_name]
    query = f"""
        SELECT
            strftime('%Y-%m', a.appointment_date) AS month,
            SUM(CASE WHEN a.appointment_status = 'Completed' THEN a.consultation_fee ELSE 0 END) AS revenue,
            AVG(CASE WHEN a.appointment_status = 'No-show' THEN 1.0 ELSE 0.0 END) AS no_show_rate,
            COUNT(*) AS appointments
        {analytics_base_sql(where_sql)}
        GROUP BY strftime('%Y-%m', a.appointment_date)
        ORDER BY month
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


@st.cache_data(show_spinner=False)
def query_booking_reference_data():
    with get_connection() as conn:
        cities = pd.read_sql_query("SELECT city_id, city_name FROM cities ORDER BY city_name", conn)
        clinics = pd.read_sql_query("SELECT clinic_id, city_id, clinic_name FROM clinics ORDER BY clinic_name", conn)
        departments = pd.read_sql_query("SELECT department_id, department_name FROM departments ORDER BY department_name", conn)
        doctors = pd.read_sql_query(
            """
            SELECT
                doctor_id,
                doctor_name,
                clinic_id,
                department_id,
                consultation_fee
            FROM doctors
            ORDER BY doctor_name
            """,
            conn,
        )
        recent = pd.read_sql_query(
            """
            SELECT
                a.appointment_scheduled_at,
                p.patient_name,
                c.city_name,
                cl.clinic_name,
                d.department_name,
                doc.doctor_name,
                a.appointment_status
            FROM appointments a
            JOIN patients p ON a.patient_id = p.patient_id
            JOIN cities c ON a.city_id = c.city_id
            JOIN clinics cl ON a.clinic_id = cl.clinic_id
            JOIN departments d ON a.department_id = d.department_id
            JOIN doctors doc ON a.doctor_id = doc.doctor_id
            ORDER BY a.appointment_id DESC
            LIMIT 8
            """,
            conn,
        )
    return cities, clinics, departments, doctors, recent


def create_patient_if_needed(conn: sqlite3.Connection, patient_name: str, city_id: int, returning_patient: bool) -> int:
    existing = conn.execute(
        """
        SELECT patient_id
        FROM patients
        WHERE lower(patient_name) = lower(?) AND city_id = ?
        ORDER BY patient_id DESC
        LIMIT 1
        """,
        (patient_name.strip(), city_id),
    ).fetchone()
    if existing and returning_patient:
        return int(existing[0])
    if existing and not returning_patient:
        return int(existing[0])

    next_id = conn.execute("SELECT COALESCE(MAX(patient_id), 0) + 1 FROM patients").fetchone()[0]
    email = f"{patient_name.strip().lower().replace(' ', '.')}.{next_id}@meditrack.local"
    phone = f"03{next_id:09d}"[:11]
    conn.execute(
        """
        INSERT INTO patients (
            patient_id,
            patient_name,
            gender,
            date_of_birth,
            city_id,
            patient_segment,
            email,
            phone,
            registered_on,
            chronic_flag
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(next_id),
            patient_name.strip(),
            "Not Specified",
            "1990-01-01",
            city_id,
            "Urban Families",
            email,
            phone,
            datetime.now().strftime("%Y-%m-%d"),
            0,
        ),
    )
    return int(next_id)


def insert_appointment(
    patient_name: str,
    city_id: int,
    clinic_id: int,
    department_id: int,
    doctor_id: int,
    scheduled_date,
    appointment_hour: int,
    returning_patient: bool,
) -> None:
    scheduled_at = datetime.combine(scheduled_date, time(appointment_hour, 0))
    booked_at = datetime.now()
    patient_type = "Returning" if returning_patient else "New"
    day_of_week = scheduled_at.strftime("%A")
    month_name = scheduled_at.strftime("%B")
    lead_days = max((scheduled_at.date() - booked_at.date()).days, 0)
    is_peak_slot = int(appointment_hour in {10, 11, 15, 16})

    with get_connection() as conn:
        doctor_row = conn.execute(
            """
            SELECT d.consultation_fee, c.city_id, doc.department_id, doc.clinic_id
            FROM doctors doc
            JOIN cities c ON c.city_id = ?
            JOIN doctors d ON d.doctor_id = doc.doctor_id
            WHERE doc.doctor_id = ?
            """,
            (city_id, doctor_id),
        ).fetchone()
        if not doctor_row:
            raise ValueError("Doctor not found for booking.")

        consultation_fee = float(doctor_row[0])
        patient_id = create_patient_if_needed(conn, patient_name, city_id, returning_patient)
        next_id = conn.execute("SELECT COALESCE(MAX(appointment_id), 0) + 1 FROM appointments").fetchone()[0]

        conn.execute(
            """
            INSERT INTO appointments (
                appointment_id,
                appointment_booked_at,
                appointment_scheduled_at,
                appointment_date,
                patient_id,
                doctor_id,
                clinic_id,
                department_id,
                city_id,
                appointment_channel,
                patient_type,
                appointment_hour,
                day_of_week,
                month_name,
                lead_days,
                consultation_fee,
                appointment_status,
                wait_time_minutes,
                consultation_duration_minutes,
                satisfaction_score,
                no_show_risk_score,
                is_peak_slot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(next_id),
                booked_at.strftime("%Y-%m-%d %H:%M:%S"),
                scheduled_at.strftime("%Y-%m-%d %H:%M:%S"),
                scheduled_at.strftime("%Y-%m-%d"),
                patient_id,
                doctor_id,
                clinic_id,
                department_id,
                city_id,
                "App",
                patient_type,
                appointment_hour,
                day_of_week,
                month_name,
                lead_days,
                consultation_fee,
                "Scheduled",
                0.0,
                0.0,
                None,
                0.12 if patient_type == "New" else 0.07,
                is_peak_slot,
            ),
        )


@st.cache_data(show_spinner=False)
def train_no_show_model(df: pd.DataFrame):
    model_df = df[df["appointment_status"] != "Scheduled"].copy()
    model_df["target"] = (model_df["appointment_status"] == "No-show").astype(int)
    features = [
        "city_name",
        "clinic_name",
        "department_name",
        "seniority_level",
        "patient_type",
        "appointment_channel",
        "patient_segment",
        "appointment_hour",
        "lead_days",
        "consultation_fee",
        "is_peak_slot",
        "years_experience",
        "popularity_score",
        "traffic_index",
        "chronic_flag",
        "day_of_week",
    ]
    X = model_df[features]
    y = model_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    categorical = [
        "city_name",
        "clinic_name",
        "department_name",
        "seniority_level",
        "patient_type",
        "appointment_channel",
        "patient_segment",
        "day_of_week",
    ]
    numeric = [
        "appointment_hour",
        "lead_days",
        "consultation_fee",
        "is_peak_slot",
        "years_experience",
        "popularity_score",
        "traffic_index",
        "chronic_flag",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=220,
                    max_depth=12,
                    min_samples_leaf=3,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importance_df = (
        pd.DataFrame(
            {"feature": feature_names, "importance": model.named_steps["classifier"].feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .head(12)
    )
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
    }
    return model, features, metrics, importance_df


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "0.0%"
    return f"{value * 100:.1f}%"


def format_pkr(value: float) -> str:
    if pd.isna(value):
        value = 0.0
    if value >= 1_000_000:
        return f"PKR {value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"PKR {value / 1_000:.1f}K"
    return f"PKR {value:.0f}"


def render_top_brand() -> None:
    st.markdown(
        """
        <div class="top-brand">
            <div style="display:flex; align-items:center; gap:0.9rem;">
                <div class="hero-mark">🏥</div>
                <div>
                    <p class="main-brand-title">MediTrack AI - Smart Healthcare Analytics</p>
                    <p class="main-brand-copy">Predictive operations, real-time appointment intelligence, and a cleaner command center for healthcare teams.</p>
                </div>
            </div>
            <div class="top-pill">AI-assisted care operations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="brand-shell">
            <div class="brand-mark">🏥</div>
            <div>
                <p class="brand-name">MediTrack AI</p>
                <p class="brand-copy">Smart Healthcare Analytics</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    render_brand()
    st.sidebar.title("🏥 MediTrack System")
    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-title">System Lens</div>
            <div class="sidebar-value">Healthcare SaaS workspace</div>
            <div style="color: rgba(255,255,255,0.72); line-height: 1.72;">
                Executive summaries, predictive no-show support, and live appointment workflows in one product.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cities, departments, min_date, max_date = get_filter_options()
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Dashboard Overview",
            "Doctors",
            "Patients",
            "Book Appointment",
            "Analytics",
        ],
    )
    st.sidebar.markdown("### Filters")
    city = st.sidebar.selectbox("City", ["All"] + cities)
    department = st.sidebar.selectbox("Department", ["All"] + departments)
    date_range = st.sidebar.date_input(
        "Date range",
        value=(pd.to_datetime(min_date).date(), pd.to_datetime(max_date).date()),
        min_value=pd.to_datetime(min_date).date(),
        max_value=pd.to_datetime(max_date).date(),
    )
    st.sidebar.markdown(
        """
        <div class="settings-panel">
            <div class="settings-title">Settings Studio</div>
            <div class="settings-copy">Switch between cleaner executive views and more guided operating views without leaving the page.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar.expander("Open Settings", expanded=False):
        preset = st.radio(
            "Mode",
            ["Balanced", "Executive", "Operations"],
            horizontal=True,
            key="settings_preset",
        )
        preset_defaults = {
            "Balanced": {
                "show_gallery": True,
                "show_booking": True,
                "show_warnings": True,
                "show_recommendations": True,
            },
            "Executive": {
                "show_gallery": True,
                "show_booking": False,
                "show_warnings": True,
                "show_recommendations": False,
            },
            "Operations": {
                "show_gallery": False,
                "show_booking": True,
                "show_warnings": True,
                "show_recommendations": True,
            },
        }
        defaults = preset_defaults[preset]
        show_gallery = st.checkbox("Show image gallery", value=defaults["show_gallery"], key="show_gallery_toggle")
        show_booking = st.checkbox("Show booking section", value=defaults["show_booking"], key="show_booking_toggle")
        show_warnings = st.checkbox("Show smart warnings", value=defaults["show_warnings"], key="show_warnings_toggle")
        show_recommendations = st.checkbox(
            "Show recommendations",
            value=defaults["show_recommendations"],
            key="show_recommendations_toggle",
        )
        enabled_count = sum([show_gallery, show_booking, show_warnings, show_recommendations])
        st.caption(f"{enabled_count} of 4 experience layers are active.")
        if st.button("Reset To Preset", key="reset_settings_preset", use_container_width=True):
            st.session_state["show_gallery_toggle"] = defaults["show_gallery"]
            st.session_state["show_booking_toggle"] = defaults["show_booking"]
            st.session_state["show_warnings_toggle"] = defaults["show_warnings"]
            st.session_state["show_recommendations_toggle"] = defaults["show_recommendations"]
            st.rerun()
    settings = {
        "show_gallery": show_gallery,
        "show_booking": show_booking,
        "show_warnings": show_warnings,
        "show_recommendations": show_recommendations,
    }
    return page, city, department, date_range, settings


def hero(page_name: str, snapshot: pd.Series) -> None:
    content = PAGE_COPY[page_name]
    image = PAGE_IMAGES[page_name][0]
    html = f"""
    <div class="hero">
        <div class="hero-grid">
            <div>
                <div class="eyebrow">{content["eyebrow"]}</div>
                <div class="hero-brand-row">
                    <div class="hero-mark">AI</div>
                    <div style="color:rgba(255,255,255,0.76); font-weight:700;">MediTrack AI workspace</div>
                </div>
                <div class="hero-title">{content["title"]}</div>
                <p class="hero-copy">{content["hook"]}</p>
                <div class="hero-mini-stats">
                    <div class="mini-stat">
                        <div class="mini-label">Appointments</div>
                        <div class="mini-value">{int(snapshot["appointments"]):,}</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Revenue</div>
                        <div class="mini-value">{format_pkr(snapshot["realized_revenue"])}</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">No-show rate</div>
                        <div class="mini-value">{format_pct(snapshot["no_show_rate"])}</div>
                    </div>
                </div>
            </div>
            <div class="hero-banner-card" title="{image["caption"]}">
                <img src="{image["url"]}" alt="{image["title"]}">
                <div class="hero-banner-caption">
                    <strong>{image["title"]}</strong><br>
                    {image["caption"]}
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(
        f'<div class="section-label">{text}</div><div class="section-rule"></div>',
        unsafe_allow_html=True,
    )


def render_auto_insight(title: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="auto-insight">
            <div class="auto-insight-title">{html.escape(title)}</div>
            <p class="auto-insight-copy">{html.escape(message)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def assistant_reply(prompt: str) -> str:
    cleaned = prompt.strip().lower()
    if "book" in cleaned and "appointment" in cleaned:
        return "Go to the Book Appointment section, choose city, clinic, department, doctor, then submit the form."
    if "use app" in cleaned or "how to use" in cleaned:
        return "Use the sidebar to switch pages, apply filters, and explore live analytics, predictions, records, and the Pakistan network map."
    if "what does this page do" in cleaned or "this page" in cleaned:
        return "This dashboard tracks healthcare performance across cities, clinics, doctors, appointments, revenue, and no-show behavior."
    if "help" in cleaned:
        return "Try asking about booking appointments, using the dashboard, viewing doctor performance, or understanding no-show insights."
    if "doctor" in cleaned:
        return "Open Doctor Explorer to compare clinicians on revenue, completion rate, no-show exposure, and patient satisfaction."
    if "no-show" in cleaned:
        return "Open No-Show Studio to review risk hotspots, timing patterns, and the prediction copilot for intervention planning."
    return "I can help with booking appointments, using the app, understanding pages, and navigating doctor or no-show analytics."


def render_assistant_panel() -> None:
    if "assistant_history" not in st.session_state:
        st.session_state["assistant_history"] = [
            {
                "role": "assistant",
                "content": "Ask me how to book appointments, use the app, or understand a dashboard section.",
            }
        ]
    st.sidebar.markdown(
        """
        <div class="assistant-shell">
            <div class="assistant-title">💬 Assistant</div>
            <div class="assistant-copy">Friendly guidance for booking, navigation, and what each page is telling you.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for message in st.session_state["assistant_history"][-4:]:
        with st.sidebar.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.sidebar.text_input("Ask the assistant", key="assistant_prompt")
    if st.sidebar.button("Send", key="assistant_send", use_container_width=True):
        if prompt.strip():
            st.session_state["assistant_history"].append({"role": "user", "content": prompt.strip()})
            st.session_state["assistant_history"].append(
                {"role": "assistant", "content": assistant_reply(prompt)}
            )
            st.rerun()


def story(text: str) -> None:
    st.markdown(
        f"""
        <div class="story-card">
            <div class="story-title">Why this matters</div>
            <p class="story-copy">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(cards: list[dict[str, str]]) -> None:
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{card["label"]}</div>
                <div class="metric-value">{card["value"]}</div>
                <div class="metric-delta">{card["delta"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def insight_grid(items: list[dict[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-tag">{item["tag"]}</div>
                <div class="insight-title">{item["title"]}</div>
                <p class="insight-body">{item["body"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_image_gallery(page_name: str) -> None:
    columns = st.columns(3)
    for col, item in zip(columns, PAGE_IMAGES[page_name]):
        title = html.escape(item["title"])
        caption = html.escape(item["caption"])
        url = html.escape(item["url"], quote=True)
        col.markdown(
            f"""
            <div class="image-card" title="{caption}">
                <img src="{url}" alt="{title}" />
                <div class="hover-arrow">Hover</div>
                <div class="hover-tooltip">
                    <div class="hover-title">{title}</div>
                    <p class="hover-copy">{caption}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_signal_board(items: list[dict[str, str]], board_type: str) -> None:
    if not items:
        return
    card_class = "warning-card" if board_type == "warning" else "recommendation-card"
    cols = st.columns(min(2, len(items)))
    for idx, item in enumerate(items):
        extra_class = " is-critical" if item.get("critical") else ""
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="{card_class}{extra_class}">
                    <div class="signal-tag">{html.escape(item["tag"])}</div>
                    <div class="signal-title">{html.escape(item["title"])}</div>
                    <p class="signal-body">{html.escape(item["body"])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_overview_warnings(snapshot: pd.Series, monthly: pd.DataFrame, city_perf: pd.DataFrame) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if float(snapshot["no_show_rate"]) >= 0.14:
        warnings.append(
            {
                "tag": "Smart Warning",
                "title": f"⚠ High no-show pressure at {format_pct(snapshot['no_show_rate'])}",
                "body": "Attendance leakage is high enough to materially suppress realized care volume across the current filtered network view.",
                "critical": True,
            }
        )
    if len(monthly) >= 2 and float(monthly["completed"].iloc[-1]) < float(monthly["completed"].iloc[-2]):
        warnings.append(
            {
                "tag": "Smart Warning",
                "title": "⚠ Completed visits softened in the latest month",
                "body": "Demand may still look healthy, but delivered consultations dropped versus the prior month and need monitoring.",
            }
        )
    if not city_perf.empty:
        fragile_city = city_perf.sort_values("no_show_rate", ascending=False).iloc[0]
        if float(fragile_city["no_show_rate"]) >= 0.12:
            warnings.append(
                {
                    "tag": "Smart Warning",
                    "title": f"⚠ {fragile_city['city_name']} is the current attendance risk pocket",
                    "body": f"No-shows are running at {format_pct(fragile_city['no_show_rate'])}, which makes this city a likely source of wasted slots.",
                }
            )
    return warnings[:4]


def build_overview_recommendations(snapshot: pd.Series, city_perf: pd.DataFrame) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    if float(snapshot["no_show_rate"]) >= 0.12:
        recommendations.append(
            {
                "tag": "Recommendation",
                "title": "Send stronger reminder flows to fragile appointments",
                "body": "Prioritize reminder plus confirmation sequences for cohorts and cities already showing elevated no-show behavior.",
            }
        )
    if not city_perf.empty:
        top_city = city_perf.sort_values("revenue", ascending=False).iloc[0]
        recommendations.append(
            {
                "tag": "Recommendation",
                "title": f"Protect capacity in {top_city['city_name']}",
                "body": "The network revenue anchor should get the cleanest schedules, fastest check-in handling, and fewer wasted slots.",
            }
        )
    return recommendations[:4]


def build_utilization_warnings(snapshot: pd.Series, dept_perf: pd.DataFrame, clinic_perf: pd.DataFrame) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if not dept_perf.empty:
        weakest_department = dept_perf.sort_values("utilization_rate").iloc[0]
        warnings.append(
            {
                "tag": "Smart Warning",
                "title": f"⚠ {weakest_department['department_name']} is underperforming this month",
                "body": f"Only {format_pct(weakest_department['utilization_rate'])} of bookings are converting into completed care in the current slice.",
                "critical": float(weakest_department["utilization_rate"]) < 0.60,
            }
        )
    if not clinic_perf.empty:
        weakest_clinic = clinic_perf.sort_values(["completion_rate", "revenue"], ascending=[True, True]).iloc[0]
        warnings.append(
            {
                "tag": "Smart Warning",
                "title": f"⚠ {weakest_clinic['clinic_name']} is leaking clinic capacity",
                "body": f"Completion sits at {format_pct(weakest_clinic['completion_rate'])} with {format_pct(weakest_clinic['no_show_rate'])} no-shows.",
            }
        )
    if not pd.isna(snapshot["avg_wait_time"]) and float(snapshot["avg_wait_time"]) >= 18:
        warnings.append(
            {
                "tag": "Smart Warning",
                "title": "⚠ Wait times are drifting into a friction zone",
                "body": f"Average completed-visit wait time is {snapshot['avg_wait_time']:.1f} minutes, which can hurt patient experience and future attendance.",
            }
        )
    return warnings[:4]


def build_utilization_recommendations(dept_perf: pd.DataFrame, clinic_perf: pd.DataFrame) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    if not dept_perf.empty:
        weakest_department = dept_perf.sort_values("utilization_rate").iloc[0]
        recommendations.append(
            {
                "tag": "Recommendation",
                "title": "Reduce slots or redesign schedules for low-conversion departments",
                "body": f"Start with {weakest_department['department_name']}, then rebalance capacity toward departments that are converting demand more cleanly.",
            }
        )
    if not clinic_perf.empty:
        busiest_good = clinic_perf.sort_values(["completion_rate", "revenue"], ascending=[False, False]).iloc[0]
        recommendations.append(
            {
                "tag": "Recommendation",
                "title": "Scale what already works operationally",
                "body": f"Replicate staffing and slot patterns from {busiest_good['clinic_name']} into weaker clinics before chasing new demand.",
            }
        )
    return recommendations[:4]


def build_no_show_warnings(snapshot: pd.Series, risk_by_type: pd.DataFrame, risk_by_hour: pd.DataFrame) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if not risk_by_hour.empty:
        peak_hour = risk_by_hour.loc[risk_by_hour["no_show_rate"].idxmax()]
        if int(peak_hour["appointment_hour"]) >= 17 or int(peak_hour["appointment_hour"]) <= 9:
            warnings.append(
                {
                    "tag": "Smart Warning",
                    "title": f"⚠ High no-show risk in evening or edge slots around {int(peak_hour['appointment_hour'])}:00",
                    "body": f"Those appointments are currently running at {format_pct(peak_hour['no_show_rate'])} no-show risk and deserve special handling.",
                    "critical": float(peak_hour["no_show_rate"]) >= 0.20,
                }
            )
    if not risk_by_type.empty:
        riskiest = risk_by_type.iloc[0]
        warnings.append(
            {
                "tag": "Smart Warning",
                "title": f"⚠ {riskiest['department_name']} has the highest fragile cohort",
                "body": f"{riskiest['patient_type']} patients are hitting {format_pct(riskiest['no_show_rate'])} no-show risk in this department.",
            }
        )
    if float(snapshot["no_show_rate"]) >= 0.13:
        warnings.append(
            {
                "tag": "Smart Warning",
                "title": "⚠ Baseline attendance risk is elevated",
                "body": "The filtered population is already running above a comfortable no-show threshold, so intervention should be proactive rather than reactive.",
            }
        )
    return warnings[:4]


def build_no_show_recommendations(risk_by_hour: pd.DataFrame, risk_by_type: pd.DataFrame) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    if not risk_by_hour.empty:
        peak_hour = risk_by_hour.loc[risk_by_hour["no_show_rate"].idxmax()]
        recommendations.append(
            {
                "tag": "Recommendation",
                "title": "Send reminders for evening appointments",
                "body": f"Start with the {int(peak_hour['appointment_hour'])}:00 slot band, where no-show behavior is currently weakest.",
            }
        )
    if not risk_by_type.empty:
        riskiest = risk_by_type.iloc[0]
        recommendations.append(
            {
                "tag": "Recommendation",
                "title": "Use backup waitlists for the riskiest specialty mix",
                "body": f"Prioritize {riskiest['department_name']} with differentiated outreach and faster slot backfilling when patients do not confirm.",
            }
        )
    return recommendations[:4]


def build_doctor_warnings(selected: pd.Series, peer_set: pd.DataFrame) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    peer_no_show_avg = float(peer_set["no_show_rate"].mean()) if not peer_set.empty else 0.0
    if float(selected["no_show_rate"]) > peer_no_show_avg:
        warnings.append(
            {
                "tag": "Smart Warning",
                "title": f"⚠ {selected['doctor_name']} is above peer no-show exposure",
                "body": f"This doctor is at {format_pct(selected['no_show_rate'])} no-show exposure versus a peer average of {format_pct(peer_no_show_avg)}.",
            }
        )
    if not pd.isna(selected["avg_satisfaction"]) and float(selected["avg_satisfaction"]) < 4.0:
        warnings.append(
            {
                "tag": "Smart Warning",
                "title": "⚠ Patient satisfaction is softer than premium-care expectations",
                "body": f"The current score is {selected['avg_satisfaction']:.2f}/5, which could weaken repeat demand quality over time.",
                "critical": True,
            }
        )
    return warnings[:4]


def build_doctor_recommendations(selected: pd.Series, top_peer: pd.Series) -> list[dict[str, str]]:
    return [
        {
            "tag": "Recommendation",
            "title": "Coach schedule design around the selected clinician",
            "body": "Shift weaker slot bands, strengthen pre-visit confirmations, and protect the hours most likely to convert into delivered care.",
        },
        {
            "tag": "Recommendation",
            "title": f"Benchmark against {top_peer['doctor_name']}",
            "body": f"Use the department leader’s throughput and attendance pattern as the practical reference point for improvement.",
        },
    ]


def render_booking_section() -> None:
    section_label("➕ Book a New Appointment")
    cities, clinics, departments, doctors, recent = query_booking_reference_data()
    st.markdown('<div class="form-shell">', unsafe_allow_html=True)
    form_cols = st.columns(4)
    with form_cols[0]:
        patient_name = st.text_input("Patient Name")
        city_name = st.selectbox("City", cities["city_name"].tolist())
        selected_city_id = int(cities.loc[cities["city_name"] == city_name, "city_id"].iloc[0])
        clinic_options = clinics[clinics["city_id"] == selected_city_id].sort_values("clinic_name")
        clinic_name = st.selectbox("Clinic", clinic_options["clinic_name"].tolist())
    with form_cols[1]:
        department_name = st.selectbox("Department", departments["department_name"].tolist())
        selected_department_id = int(
            departments.loc[departments["department_name"] == department_name, "department_id"].iloc[0]
        )
        selected_clinic_id = int(
            clinic_options.loc[clinic_options["clinic_name"] == clinic_name, "clinic_id"].iloc[0]
        )
        doctor_options = doctors[
            (doctors["clinic_id"] == selected_clinic_id) & (doctors["department_id"] == selected_department_id)
        ].sort_values("doctor_name")
        if doctor_options.empty:
            doctor_options = doctors[doctors["clinic_id"] == selected_clinic_id].sort_values("doctor_name")
        doctor_name = st.selectbox("Doctor", doctor_options["doctor_name"].tolist())
    with form_cols[2]:
        selected_date = st.date_input("Date", value=datetime.now().date())
        hour = st.slider("Time", 9, 18, 15)
        returning_patient = st.checkbox("Returning patient")
    with form_cols[3]:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"**Consultation fee:** {format_pkr(float(doctor_options.loc[doctor_options['doctor_name'] == doctor_name, 'consultation_fee'].iloc[0]))}"
        )
        submit = st.button("Book Appointment", use_container_width=True)

    if submit:
        if not patient_name.strip():
            st.error("Patient name is required.")
        else:
            doctor_id = int(doctor_options.loc[doctor_options["doctor_name"] == doctor_name, "doctor_id"].iloc[0])
            try:
                insert_appointment(
                    patient_name=patient_name,
                    city_id=selected_city_id,
                    clinic_id=selected_clinic_id,
                    department_id=selected_department_id,
                    doctor_id=doctor_id,
                    scheduled_date=selected_date,
                    appointment_hour=hour,
                    returning_patient=returning_patient,
                )
                st.cache_data.clear()
                st.success("Appointment booked successfully!")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not book appointment: {exc}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Latest booked records**")
    if not recent.empty:
        st.dataframe(recent, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def query_record_reference_data():
    with get_connection() as conn:
        cities = pd.read_sql_query(
            "SELECT city_id, city_name FROM cities ORDER BY city_name",
            conn,
        )
        clinics = pd.read_sql_query(
            "SELECT clinic_id, city_id, clinic_name FROM clinics ORDER BY clinic_name",
            conn,
        )
        departments = pd.read_sql_query(
            "SELECT department_id, department_name FROM departments ORDER BY department_name",
            conn,
        )
    return cities, clinics, departments


def insert_patient_record(patient_name: str, city_id: int, is_returning: bool) -> None:
    clean_name = patient_name.strip()
    with get_connection() as conn:
        next_id = conn.execute("SELECT COALESCE(MAX(patient_id), 0) + 1 FROM patients").fetchone()[0]
        email = f"{clean_name.lower().replace(' ', '.')}.{int(next_id)}@meditrack.local"
        phone = f"03{int(next_id):09d}"[:11]
        patient_segment = "Returning Care" if is_returning else "Urban Families"
        conn.execute(
            """
            INSERT INTO patients (
                patient_id,
                patient_name,
                gender,
                date_of_birth,
                city_id,
                patient_segment,
                email,
                phone,
                registered_on,
                chronic_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(next_id),
                clean_name,
                "Not Specified",
                "1990-01-01",
                city_id,
                patient_segment,
                email,
                phone,
                datetime.now().strftime("%Y-%m-%d"),
                0,
            ),
        )


def insert_doctor_record(
    doctor_name: str,
    clinic_id: int,
    department_id: int,
    seniority_level: str,
    consultation_fee: float,
) -> None:
    years_lookup = {"Junior": 2, "Mid-level": 6, "Senior": 12}
    popularity_lookup = {"Junior": 68.0, "Mid-level": 79.0, "Senior": 91.0}
    with get_connection() as conn:
        next_id = conn.execute("SELECT COALESCE(MAX(doctor_id), 0) + 1 FROM doctors").fetchone()[0]
        conn.execute(
            """
            INSERT INTO doctors (
                doctor_id,
                clinic_id,
                department_id,
                doctor_name,
                gender,
                seniority_level,
                years_experience,
                consultation_fee,
                popularity_score,
                join_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(next_id),
                clinic_id,
                department_id,
                doctor_name.strip(),
                "Not Specified",
                seniority_level,
                years_lookup[seniority_level],
                float(consultation_fee),
                popularity_lookup[seniority_level],
                datetime.now().strftime("%Y-%m-%d"),
            ),
        )


@st.cache_data(show_spinner=False)
def query_pakistan_network_overview() -> dict[str, pd.DataFrame]:
    tracked_cities = ["Karachi", "Lahore", "Islamabad", "Peshawar", "Multan"]
    coords = {
        "Karachi": {"lat": 24.8607, "lon": 67.0011},
        "Lahore": {"lat": 31.5204, "lon": 74.3587},
        "Islamabad": {"lat": 33.6844, "lon": 73.0479},
        "Peshawar": {"lat": 34.0151, "lon": 71.5249},
        "Multan": {"lat": 30.1575, "lon": 71.5249},
    }
    placeholders = ", ".join("?" for _ in tracked_cities)
    with get_connection() as conn:
        cities_df = pd.read_sql_query(
            f"""
            SELECT
                c.city_id,
                c.city_name,
                (
                    SELECT COUNT(*)
                    FROM clinics cl
                    WHERE cl.city_id = c.city_id
                ) AS clinics_count,
                (
                    SELECT COUNT(*)
                    FROM doctors doc
                    JOIN clinics cl ON doc.clinic_id = cl.clinic_id
                    WHERE cl.city_id = c.city_id
                ) AS doctors_count,
                (
                    SELECT COUNT(*)
                    FROM patients p
                    WHERE p.city_id = c.city_id
                ) AS patients_count,
                (
                    SELECT COUNT(*)
                    FROM appointments a
                    WHERE a.city_id = c.city_id
                ) AS appointments,
                (
                    SELECT COALESCE(SUM(a.consultation_fee), 0)
                    FROM appointments a
                    WHERE a.city_id = c.city_id
                      AND a.appointment_status = 'Completed'
                ) AS revenue
            FROM cities c
            WHERE c.city_name IN ({placeholders})
            ORDER BY revenue DESC, c.city_name
            """,
            conn,
            params=tracked_cities,
        )
        clinics_df = pd.read_sql_query(
            f"""
            SELECT
                cl.clinic_id,
                cl.clinic_name AS name,
                c.city_name AS city,
                COALESCE(SUM(CASE WHEN a.appointment_status = 'Completed' THEN a.consultation_fee ELSE 0 END), 0) AS revenue,
                AVG(CASE WHEN a.appointment_status = 'No-show' THEN 1.0 ELSE 0.0 END) AS no_show_rate,
                COUNT(a.appointment_id) AS appointments
            FROM clinics cl
            JOIN cities c ON cl.city_id = c.city_id
            LEFT JOIN appointments a ON cl.clinic_id = a.clinic_id
            WHERE c.city_name IN ({placeholders})
            GROUP BY cl.clinic_id, cl.clinic_name, c.city_name
            ORDER BY revenue DESC, cl.clinic_name
            """,
            conn,
            params=tracked_cities,
        )
        doctors_df = pd.read_sql_query(
            f"""
            SELECT
                doc.doctor_id,
                doc.doctor_name AS name,
                d.department_name AS department,
                c.city_name AS city,
                AVG(CASE WHEN a.appointment_status = 'No-show' THEN 1.0 ELSE 0.0 END) AS no_show
            FROM doctors doc
            JOIN clinics cl ON doc.clinic_id = cl.clinic_id
            JOIN cities c ON cl.city_id = c.city_id
            JOIN departments d ON doc.department_id = d.department_id
            LEFT JOIN appointments a ON doc.doctor_id = a.doctor_id
            WHERE c.city_name IN ({placeholders})
            GROUP BY doc.doctor_id, doc.doctor_name, d.department_name, c.city_name
            ORDER BY doc.doctor_name
            """,
            conn,
            params=tracked_cities,
        )
    if cities_df.empty:
        return {"cities": cities_df, "clinics": clinics_df, "doctors": doctors_df}

    cities_df["lat"] = cities_df["city_name"].map(lambda city: coords[city]["lat"])
    cities_df["lon"] = cities_df["city_name"].map(lambda city: coords[city]["lon"])
    cities_df["marker_size"] = cities_df["doctors_count"].clip(lower=1) * 6

    if not clinics_df.empty:
        clinics_df["lat"] = clinics_df["city"].map(lambda city: coords[city]["lat"])
        clinics_df["lon"] = clinics_df["city"].map(lambda city: coords[city]["lon"])
        clinic_angles = np.linspace(0, 2 * np.pi, len(clinics_df), endpoint=False)
        clinic_offsets = 0.12 + (clinics_df.index.to_series() % 4) * 0.03
        clinics_df["lat"] = clinics_df["lat"] + np.sin(clinic_angles) * clinic_offsets
        clinics_df["lon"] = clinics_df["lon"] + np.cos(clinic_angles) * clinic_offsets
        clinics_df["no_show_rate"] = clinics_df["no_show_rate"].fillna(0.0) * 100

    if not doctors_df.empty:
        doctors_df["lat"] = doctors_df["city"].map(lambda city: coords[city]["lat"])
        doctors_df["lon"] = doctors_df["city"].map(lambda city: coords[city]["lon"])
        doctor_angles = np.linspace(0, 4 * np.pi, len(doctors_df), endpoint=False)
        doctor_offsets = 0.05 + (doctors_df.index.to_series() % 5) * 0.015
        doctors_df["lat"] = doctors_df["lat"] + np.sin(doctor_angles) * doctor_offsets
        doctors_df["lon"] = doctors_df["lon"] + np.cos(doctor_angles) * doctor_offsets
        doctors_df["no_show"] = doctors_df["no_show"].fillna(0.0) * 100

    return {"cities": cities_df, "clinics": clinics_df, "doctors": doctors_df}


@st.cache_data(show_spinner=False)
def query_top_performance_summary() -> dict[str, str]:
    with get_connection() as conn:
        top_city = conn.execute(
            """
            SELECT
                c.city_name,
                COALESCE(SUM(CASE WHEN a.appointment_status = 'Completed' THEN a.consultation_fee ELSE 0 END), 0) AS revenue
            FROM cities c
            LEFT JOIN appointments a ON c.city_id = a.city_id
            GROUP BY c.city_id, c.city_name
            ORDER BY revenue DESC, c.city_name
            LIMIT 1
            """
        ).fetchone()
        best_doctor = conn.execute(
            """
            SELECT
                doc.doctor_name,
                SUM(CASE WHEN a.appointment_status = 'Completed' THEN 1 ELSE 0 END) AS completed_count,
                AVG(CASE WHEN a.appointment_status = 'No-show' THEN 1.0 ELSE 0.0 END) AS no_show_rate
            FROM doctors doc
            LEFT JOIN appointments a ON doc.doctor_id = a.doctor_id
            GROUP BY doc.doctor_id, doc.doctor_name
            ORDER BY completed_count DESC, no_show_rate ASC, doc.doctor_name
            LIMIT 1
            """
        ).fetchone()
        top_department = conn.execute(
            """
            SELECT
                d.department_name,
                COUNT(a.appointment_id) AS activity_count
            FROM departments d
            LEFT JOIN appointments a ON d.department_id = a.department_id
            GROUP BY d.department_id, d.department_name
            ORDER BY activity_count DESC, d.department_name
            LIMIT 1
            """
        ).fetchone()
    return {
        "top_city": top_city[0] if top_city else "n/a",
        "top_city_revenue": format_pkr(float(top_city[1])) if top_city else "PKR 0",
        "best_doctor": best_doctor[0] if best_doctor else "n/a",
        "best_doctor_value": f"{int(best_doctor[1])} completed" if best_doctor else "0 completed",
        "top_department": top_department[0] if top_department else "n/a",
        "top_department_value": f"{int(top_department[1])} appointments" if top_department else "0 appointments",
    }


@st.cache_data(show_spinner=False)
def query_city_detail_summary(city_name: str) -> dict[str, str]:
    with get_connection() as conn:
        city_detail = conn.execute(
            """
            SELECT
                c.city_name,
                (
                    SELECT COUNT(*)
                    FROM doctors doc
                    JOIN clinics cl ON doc.clinic_id = cl.clinic_id
                    WHERE cl.city_id = c.city_id
                ) AS doctors_count,
                (
                    SELECT COUNT(*)
                    FROM patients p
                    WHERE p.city_id = c.city_id
                ) AS patients_count,
                (
                    SELECT COALESCE(SUM(a.consultation_fee), 0)
                    FROM appointments a
                    WHERE a.city_id = c.city_id
                      AND a.appointment_status = 'Completed'
                ) AS revenue,
                (
                    SELECT d.department_name
                    FROM departments d
                    JOIN appointments a ON d.department_id = a.department_id
                    WHERE a.city_id = c.city_id
                    GROUP BY d.department_id, d.department_name
                    ORDER BY COUNT(a.appointment_id) DESC, d.department_name
                    LIMIT 1
                ) AS best_department
            FROM cities c
            WHERE c.city_name = ?
            LIMIT 1
            """,
            (city_name,),
        ).fetchone()
    if not city_detail:
        return {}
    return {
        "city_name": city_detail[0],
        "doctors_count": f"{int(city_detail[1])}",
        "patients_count": f"{int(city_detail[2])}",
        "revenue": format_pkr(float(city_detail[3])),
        "best_department": city_detail[4] or "n/a",
    }


@st.cache_data(show_spinner=False)
def query_patient_management_data(city: str, department: str, start_date: str, end_date: str) -> pd.DataFrame:
    where_sql, params = build_where_clause(city, department, start_date, end_date)
    query = f"""
        SELECT
            p.patient_id,
            p.patient_name,
            p.date_of_birth,
            MAX(DATE(a.appointment_date)) AS last_appointment,
            SUM(CASE WHEN a.patient_type = 'New' THEN 1 ELSE 0 END) AS new_visits,
            SUM(CASE WHEN a.patient_type = 'Returning' THEN 1 ELSE 0 END) AS returning_visits,
            SUM(CASE WHEN a.appointment_status = 'No-show' THEN 1 ELSE 0 END) AS no_show_count,
            COUNT(a.appointment_id) AS total_appointments
        {analytics_base_sql(where_sql)}
        GROUP BY p.patient_id, p.patient_name, p.date_of_birth
        ORDER BY last_appointment DESC, p.patient_name
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return df
    dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
    today = pd.Timestamp(datetime.now().date())
    df["age"] = ((today - dob).dt.days / 365.25).fillna(0).astype(int)
    df["patient_type_label"] = np.where(df["returning_visits"] > 0, "Returning", "New")
    df["high_risk_flag"] = np.where(
        (df["no_show_count"] >= 2) | ((df["no_show_count"] / df["total_appointments"].replace(0, np.nan)) >= 0.34),
        "High Risk",
        "Stable",
    )
    return df


@st.cache_data(show_spinner=False)
def query_booking_module_data():
    with get_connection() as conn:
        patients = pd.read_sql_query(
            """
            SELECT patient_id, patient_name, city_id
            FROM patients
            ORDER BY patient_name
            """,
            conn,
        )
        cities = pd.read_sql_query("SELECT city_id, city_name FROM cities ORDER BY city_name", conn)
        clinics = pd.read_sql_query("SELECT clinic_id, city_id, clinic_name FROM clinics ORDER BY clinic_name", conn)
        departments = pd.read_sql_query("SELECT department_id, department_name FROM departments ORDER BY department_name", conn)
        doctors = pd.read_sql_query(
            """
            SELECT doctor_id, doctor_name, clinic_id, department_id, consultation_fee
            FROM doctors
            ORDER BY doctor_name
            """,
            conn,
        )
    return patients, cities, clinics, departments, doctors


def insert_existing_patient_appointment(
    patient_id: int,
    doctor_id: int,
    clinic_id: int,
    department_id: int,
    city_id: int,
    scheduled_date,
    appointment_hour: int,
) -> None:
    scheduled_at = datetime.combine(scheduled_date, time(appointment_hour, 0))
    booked_at = datetime.now()
    with get_connection() as conn:
        consultation_fee = conn.execute(
            "SELECT consultation_fee FROM doctors WHERE doctor_id = ?",
            (doctor_id,),
        ).fetchone()
        if not consultation_fee:
            raise ValueError("Doctor not found.")
        previous_visits = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()[0]
        patient_type = "Returning" if int(previous_visits) > 0 else "New"
        next_id = conn.execute("SELECT COALESCE(MAX(appointment_id), 0) + 1 FROM appointments").fetchone()[0]
        conn.execute(
            """
            INSERT INTO appointments (
                appointment_id,
                appointment_booked_at,
                appointment_scheduled_at,
                appointment_date,
                patient_id,
                doctor_id,
                clinic_id,
                department_id,
                city_id,
                appointment_channel,
                patient_type,
                appointment_hour,
                day_of_week,
                month_name,
                lead_days,
                consultation_fee,
                appointment_status,
                wait_time_minutes,
                consultation_duration_minutes,
                satisfaction_score,
                no_show_risk_score,
                is_peak_slot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(next_id),
                booked_at.strftime("%Y-%m-%d %H:%M:%S"),
                scheduled_at.strftime("%Y-%m-%d %H:%M:%S"),
                scheduled_at.strftime("%Y-%m-%d"),
                patient_id,
                doctor_id,
                clinic_id,
                department_id,
                city_id,
                "App",
                patient_type,
                appointment_hour,
                scheduled_at.strftime("%A"),
                scheduled_at.strftime("%B"),
                max((scheduled_at.date() - booked_at.date()).days, 0),
                float(consultation_fee[0]),
                "Scheduled",
                0.0,
                0.0,
                None,
                0.08 if patient_type == "Returning" else 0.12,
                int(appointment_hour in {10, 11, 15, 16}),
            ),
        )


@st.cache_data(show_spinner=False)
def query_doctor_weekly(doctor_name: str, city: str, department: str, start_date: str, end_date: str) -> pd.DataFrame:
    where_sql, params = build_where_clause(city, department, start_date, end_date)
    where_sql = f"{where_sql} AND doc.doctor_name = ?"
    params = params + [doctor_name]
    query = f"""
        SELECT
            a.day_of_week,
            COUNT(*) AS appointments
        {analytics_base_sql(where_sql)}
        GROUP BY a.day_of_week
    """
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return df
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=day_order, ordered=True)
    return df.sort_values("day_of_week")


def render_module_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="story-card" style="margin-top:0.2rem;">
            <div class="story-title">{html.escape(title)}</div>
            <p class="story-copy">{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def module_chat_reply(current_page: str, prompt: str) -> str:
    cleaned = prompt.strip().lower()
    page_context = {
        "Dashboard Overview": "You are in Dashboard Overview, where leadership monitors revenue, appointments, and attendance health.",
        "Doctors": "You are in Doctor Management section, where clinicians can be filtered, compared, and reviewed.",
        "Patients": "You are viewing Patient records, including risk flags and appointment history.",
        "Book Appointment": "You are in Book Appointment. Fill the form to schedule an appointment for an existing patient.",
        "Analytics": "You are in Analytics, where city, department, monthly, and no-show trends are explored.",
    }
    app_map = {
        "dashboard": "Dashboard Overview shows the executive front page, key KPIs, and the Pakistan network overview.",
        "overview": "Dashboard Overview is the executive landing page with appointments, revenue, no-show rate, and the network map.",
        "doctors": "The Doctors module lets you search doctors, filter by department, open a profile, and review revenue, weekly demand, and no-show patterns.",
        "patients": "The Patients module shows patient records, distinguishes new versus returning patients, and highlights higher-risk no-show behavior.",
        "book": "The Book Appointment page is used to choose a patient, doctor, department, city, clinic, date, and time slot before confirming.",
        "appointment": "To book an appointment, open Book Appointment, select the patient and doctor, then confirm the slot.",
        "analytics": "The Analytics page gathers revenue by city, department performance, monthly volume, no-show analysis, and day-of-week patterns in one place.",
        "map": "The Pakistan map shows clinic and city-level network performance so you can understand where activity and revenue are concentrated.",
        "settings": "The settings panel lets you switch between Balanced, Executive, and Operations modes, then manually turn gallery, booking, warnings, and recommendations on or off.",
        "filters": "Use the sidebar filters to narrow the app by city, department, and date range. The visible dashboards and modules update from that selection.",
        "revenue": "Revenue in MediTrack reflects completed consultation fees, so it tracks monetized care rather than raw bookings.",
        "no-show": "No-show rate measures how often appointments are missed. Higher values mean schedule waste, weaker utilization, and lost revenue.",
        "profile": "In Doctors, click View Profile to open a doctor-level breakdown with revenue trend, weekly demand, patient volume, and no-show rate.",
    }

    if not cleaned:
        return page_context[current_page]
    if "help" in cleaned:
        return page_context[current_page] + " Ask about booking, doctors, patients, analytics, the map, filters, or settings."
    if "what is this" in cleaned or "how to use" in cleaned or "how do i use" in cleaned:
        return page_context[current_page]
    if "where am i" in cleaned or "current page" in cleaned:
        return page_context[current_page]
    if "book appointment" in cleaned:
        return "Open the Book Appointment page, choose patient, doctor, department, clinic, date, and time slot, then confirm."
    if "view profile" in cleaned:
        return "Go to Doctors, find a clinician card, then click View Profile to open charts and KPI details for that doctor."
    if "city" in cleaned and "filter" in cleaned:
        return "Use the City filter in the sidebar to narrow all modules and dashboards to one market or keep it on All for the full network."
    if "department" in cleaned and "filter" in cleaned:
        return "Use the Department filter in the sidebar to focus the dashboards, doctors list, patient table, and analytics on one specialty."

    for keyword, answer in app_map.items():
        if keyword in cleaned:
            return answer

    return (
        page_context[current_page]
        + " I can answer questions about the dashboard, doctors, patients, booking, analytics, filters, the map, no-show rate, and settings."
    )


def render_floating_chatbot(current_page: str) -> None:
    if "chat_open" not in st.session_state:
        st.session_state["chat_open"] = False
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": module_chat_reply(current_page, "help"),
            }
        ]

    if not st.session_state["chat_open"]:
        if st.button("💬", key="floating_chat_open"):
            st.session_state["chat_open"] = True
            st.rerun()
        return

    rendered_messages = []
    for message in st.session_state["messages"][-8:]:
        role_class = "user" if message["role"] == "user" else "bot"
        rendered_messages.append(
            f'<div class="chat-row {role_class}"><div class="chat-bubble">{html.escape(message["content"])}</div></div>'
        )

    st.markdown(
        f"""
        <div class="floating-chat-panel">
            <div class="floating-chat-header">
                Assistant 🤖
                <div class="floating-chat-subtitle">{html.escape(module_chat_reply(current_page, "what is this"))}</div>
            </div>
            <div class="floating-chat-body">
                {''.join(rendered_messages)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("✖", key="floating_chat_close"):
        st.session_state["chat_open"] = False
        st.rerun()
    prompt = st.text_input("Message assistant", key="floating_chat_input", label_visibility="collapsed", value=st.session_state.get("floating_chat_input", ""))
    if st.button("Send", key="floating_chat_send"):
        if prompt.strip():
            st.session_state["messages"].append({"role": "user", "content": prompt.strip()})
            st.session_state["messages"].append(
                {"role": "assistant", "content": module_chat_reply(current_page, prompt)}
            )
            st.session_state["floating_chat_input"] = ""
            st.rerun()


def saas_dashboard_overview(city: str, department: str, start_date: str, end_date: str, settings: dict[str, bool]) -> None:
    snapshot = query_snapshot(city, department, start_date, end_date).iloc[0]
    city_perf = query_city_metrics(city, department, start_date, end_date)
    detail_df = load_detail_data(city, department, start_date, end_date)
    hero("Network Overview", snapshot)
    st.markdown("<br>", unsafe_allow_html=True)
    render_module_header(
        "📊 Dashboard Overview",
        "Executive command view for appointments, revenue, no-show pressure, and network-level operating health.",
    )
    if settings["show_gallery"]:
        render_image_gallery("Network Overview")
    st.markdown("<br>", unsafe_allow_html=True)
    metric_cards(
        [
            {"label": "📅 Total Appointments", "value": f"{int(snapshot['appointments']):,}", "delta": "All appointments in the current filter context."},
            {"label": "💰 Revenue", "value": format_pkr(snapshot["realized_revenue"]), "delta": "Completed-visit revenue captured across the selected slice."},
            {"label": "⚠️ No-show Rate", "value": format_pct(snapshot["no_show_rate"]), "delta": "Missed appointments as a share of total scheduled demand."},
        ]
    )
    st.markdown("<br>", unsafe_allow_html=True)
    render_pakistan_network_overview_section()
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        fig_city = px.bar(
            city_perf.sort_values("revenue", ascending=False),
            x="city_name",
            y="revenue",
            color="revenue",
            color_continuous_scale=["#bfdbfe", "#22c55e", "#7c3aed"],
            title="Revenue by city",
        )
        fig_city = chart_theme(fig_city)
        st.plotly_chart(fig_city, use_container_width=True)
        if not city_perf.empty:
            winner = city_perf.sort_values("revenue", ascending=False).iloc[0]
            st.success(f"🏆 {winner['city_name']} is the top revenue generating city.")
    with right:
        status_df = detail_df.groupby("appointment_status", as_index=False).size()
        fig_status = px.pie(
            status_df,
            names="appointment_status",
            values="size",
            color="appointment_status",
            color_discrete_map={
                "Completed": BRAND["green"],
                "Cancelled": BRAND["amber"],
                "No-show": BRAND["rose"],
                "Scheduled": BRAND["blue"],
            },
            title="Appointments by status",
            hole=0.48,
        )
        fig_status = chart_theme(fig_status)
        st.plotly_chart(fig_status, use_container_width=True)
        if float(snapshot["no_show_rate"]) > 0.25:
            st.warning("⚠ High no-show rate detected in the current dashboard view.")


def saas_doctors_page(city: str, department: str, start_date: str, end_date: str, settings: dict[str, bool]) -> None:
    doctor_perf = query_doctor_metrics(city, department, start_date, end_date)
    render_module_header(
        "👨‍⚕️ Doctors",
        "Search clinicians, filter by specialty, and open focused performance profiles without leaving the management workspace.",
    )
    if settings["show_gallery"]:
        render_image_gallery("Doctor Explorer")
    st.markdown("<br>", unsafe_allow_html=True)
    search = st.text_input("Search doctors", key="doctor_search")
    dept_options = ["All"] + sorted(doctor_perf["department_name"].dropna().unique().tolist())
    dept_filter = st.selectbox("Filter by department", dept_options, key="doctor_module_department")
    filtered = doctor_perf.copy()
    if search.strip():
        filtered = filtered[filtered["doctor_name"].str.contains(search.strip(), case=False, na=False)]
    if dept_filter != "All":
        filtered = filtered[filtered["department_name"] == dept_filter]

    selected_doctor = st.session_state.get("selected_doctor_profile")
    selected_profile_df = doctor_perf[doctor_perf["doctor_name"] == selected_doctor] if selected_doctor else pd.DataFrame()

    if filtered.empty:
        st.warning("No doctors matched the current search and filter combination.")
    else:
        for _, row in filtered.head(12).iterrows():
            card_cols = st.columns([4.3, 1])
            with card_cols[0]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">👨‍⚕️ {html.escape(row['doctor_name'])}</div>
                        <div class="metric-value">{html.escape(row['department_name'])}</div>
                        <div class="metric-delta">
                            {html.escape(row['seniority_level'])} · Revenue {format_pkr(row['revenue'])} · No-show {format_pct(row['no_show_rate'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with card_cols[1]:
                if st.button("View Profile", key=f"view_profile_{row['doctor_name']}"):
                    st.session_state["selected_doctor_profile"] = row["doctor_name"]
                    st.rerun()

    if not selected_profile_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        section_label(f"🩺 {selected_doctor} Profile")
        selected = selected_profile_df.iloc[0]
        doctor_monthly = query_doctor_monthly(selected_doctor, city, department, start_date, end_date)
        doctor_weekly = query_doctor_weekly(selected_doctor, city, department, start_date, end_date)
        total_patients = int(
            load_detail_data(city, department, start_date, end_date)
            .query("doctor_name == @selected_doctor")["appointment_id"]
            .count()
        )
        metric_cards(
            [
                {"label": "💰 Total Revenue", "value": format_pkr(selected["revenue"]), "delta": "Completed-care revenue for this doctor."},
                {"label": "✅ Completion Rate", "value": format_pct(selected["completion_rate"]), "delta": "Share of appointments that became delivered care."},
                {"label": "👥 Total Patients Handled", "value": f"{total_patients:,}", "delta": "Appointment count handled in the selected date range."},
                {"label": "⚠️ No-show Rate", "value": format_pct(selected["no_show_rate"]), "delta": "Attendance fragility associated with this doctor."},
            ]
        )
        left, right = st.columns(2)
        with left:
            fig_trend = px.line(
                doctor_monthly,
                x="month",
                y="revenue",
                markers=True,
                color_discrete_sequence=[BRAND["blue"]],
                title="Revenue trend",
            )
            fig_trend = chart_theme(fig_trend)
            st.plotly_chart(fig_trend, use_container_width=True)
            st.info(f"👨‍⚕️ {selected_doctor} has the highest visibility in this profile view.")
        with right:
            fig_weekly = px.bar(
                doctor_weekly,
                x="day_of_week",
                y="appointments",
                color="appointments",
                color_continuous_scale=["#bfdbfe", "#14b8a6", "#7c3aed"],
                title="Weekly appointment trend",
            )
            fig_weekly = chart_theme(fig_weekly)
            st.plotly_chart(fig_weekly, use_container_width=True)
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=float(selected["no_show_rate"]) * 100,
                    number={"suffix": "%"},
                    title={"text": "No-show rate"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": BRAND["rose"]},
                        "steps": [
                            {"range": [0, 10], "color": "#dcfce7"},
                            {"range": [10, 20], "color": "#fef3c7"},
                            {"range": [20, 100], "color": "#fee2e2"},
                        ],
                    },
                )
            )
            gauge = chart_theme(gauge)
            st.plotly_chart(gauge, use_container_width=True)


def saas_patients_page(city: str, department: str, start_date: str, end_date: str, settings: dict[str, bool]) -> None:
    patients_df = query_patient_management_data(city, department, start_date, end_date)
    render_module_header(
        "🧑 Patients",
        "Patient record management with new versus returning filters, recent activity, and automatic no-show risk flagging.",
    )
    if settings["show_gallery"]:
        render_image_gallery("Network Overview")
    st.markdown("<br>", unsafe_allow_html=True)
    if patients_df.empty:
        st.warning("No patient records are available for the selected filter set.")
        return
    patient_filter = st.selectbox("Patient filter", ["All", "New patients", "Returning patients"], key="patient_filter")
    filtered = patients_df.copy()
    if patient_filter == "New patients":
        filtered = filtered[filtered["patient_type_label"] == "New"]
    elif patient_filter == "Returning patients":
        filtered = filtered[filtered["patient_type_label"] == "Returning"]
    high_risk = filtered[filtered["high_risk_flag"] == "High Risk"]
    if not high_risk.empty:
        st.warning(f"⚠ {len(high_risk)} patients are flagged as high-risk due to frequent no-shows.")
    display_df = filtered[
        ["patient_id", "patient_name", "age", "patient_type_label", "last_appointment", "high_risk_flag"]
    ].rename(
        columns={
            "patient_id": "Patient ID",
            "patient_name": "Name",
            "age": "Age",
            "patient_type_label": "New / Returning",
            "last_appointment": "Last appointment",
            "high_risk_flag": "Risk",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def saas_book_appointment_page(settings: dict[str, bool]) -> None:
    patients, cities, clinics, departments, doctors = query_booking_module_data()
    render_module_header(
        "📅 Book Appointment",
        "Schedule appointments with a cleaner operational form built for front-desk speed and healthcare workflow clarity.",
    )
    if settings["show_gallery"]:
        render_image_gallery("Network Overview")
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        patient_name = st.text_input("Patient Name", key="saas_patient")
        if patient_name.strip() and patient_name not in patients["patient_name"].tolist():
            st.error("Patient not found. Please select from existing patients or add a new one in Manage Records.")
            patient_id = None
        elif patient_name.strip():
            patient_id = int(patients.loc[patients["patient_name"] == patient_name, "patient_id"].iloc[0])
        else:
            patient_id = None
        city_name = st.selectbox("Select City", cities["city_name"].tolist(), key="saas_city")
        city_id = int(cities.loc[cities["city_name"] == city_name, "city_id"].iloc[0])
        clinic_options = clinics[clinics["city_id"] == city_id].sort_values("clinic_name")
        clinic_name = st.selectbox("Select Clinic", clinic_options["clinic_name"].tolist(), key="saas_clinic")
        clinic_id = int(clinic_options.loc[clinic_options["clinic_name"] == clinic_name, "clinic_id"].iloc[0])
    with col2:
        department_name = st.selectbox("Select Department", departments["department_name"].tolist(), key="saas_department")
        department_id = int(
            departments.loc[departments["department_name"] == department_name, "department_id"].iloc[0]
        )
        doctor_options = doctors[
            (doctors["clinic_id"] == clinic_id) & (doctors["department_id"] == department_id)
        ].sort_values("doctor_name")
        if doctor_options.empty:
            doctor_options = doctors[doctors["clinic_id"] == clinic_id].sort_values("doctor_name")
        doctor_name = st.selectbox("Select Doctor", doctor_options["doctor_name"].tolist(), key="saas_doctor")
        doctor_id = int(doctor_options.loc[doctor_options["doctor_name"] == doctor_name, "doctor_id"].iloc[0])
    with col3:
        selected_date = st.date_input("Select Date", value=datetime.now().date(), key="saas_date")
        appointment_hour = st.selectbox("Select Time slot", list(range(9, 19)), key="saas_hour")
        st.markdown(
            f"**Consultation fee:** {format_pkr(float(doctor_options.loc[doctor_options['doctor_name'] == doctor_name, 'consultation_fee'].iloc[0]))}"
        )
    if st.button("Confirm Appointment", key="saas_confirm_appointment", use_container_width=True):
        if patient_id is None:
            st.error("Please enter a valid patient name.")
        else:
            insert_existing_patient_appointment(
                patient_id=patient_id,
                doctor_id=doctor_id,
                clinic_id=clinic_id,
                department_id=department_id,
                city_id=city_id,
                scheduled_date=selected_date,
                appointment_hour=int(appointment_hour),
            )
            st.session_state.setdefault("booked_appointments", []).append(
                {
                    "patient": patient_name,
                    "doctor": doctor_name,
                    "department": department_name,
                    "city": city_name,
                    "clinic": clinic_name,
                    "date": str(selected_date),
                    "time": f"{appointment_hour}:00",
                }
            )
            st.cache_data.clear()
            st.success("Appointment confirmed and stored successfully.")
            st.rerun()
    if st.session_state.get("booked_appointments"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state["booked_appointments"]), use_container_width=True, hide_index=True)


def saas_analytics_page(city: str, department: str, start_date: str, end_date: str, settings: dict[str, bool]) -> None:
    render_module_header(
        "📈 Analytics",
        "A focused analytics workspace for city revenue, department efficiency, appointment momentum, no-show behavior, and weekly patterns.",
    )
    if settings["show_gallery"]:
        render_image_gallery("Utilization & Revenue")
    st.markdown("<br>", unsafe_allow_html=True)
    city_perf = query_city_metrics(city, department, start_date, end_date)
    dept_perf = query_department_metrics(city, department, start_date, end_date)
    monthly = query_monthly_metrics(city, department, start_date, end_date)
    risk_by_type = query_no_show_by_type(city, department, start_date, end_date)
    detail_df = load_detail_data(city, department, start_date, end_date)
    day_of_week = (
        detail_df.groupby("day_of_week", as_index=False)
        .agg(appointments=("appointment_id", "count"), no_show_rate=("is_no_show", "mean"))
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_of_week["day_of_week"] = pd.Categorical(day_of_week["day_of_week"], categories=day_order, ordered=True)
    day_of_week = day_of_week.sort_values("day_of_week")

    tab1, tab2, tab3 = st.tabs(["Revenue", "Operations", "Attendance"])
    with tab1:
        fig_city = px.bar(
            city_perf.sort_values("revenue", ascending=False),
            x="city_name",
            y="revenue",
            color="revenue",
            color_continuous_scale=["#bfdbfe", "#14b8a6", "#7c3aed"],
            title="Revenue by city",
        )
        fig_city = chart_theme(fig_city)
        st.plotly_chart(fig_city, use_container_width=True)
        if not city_perf.empty:
            st.success(f"🏆 {city_perf.iloc[0]['city_name']} is the top revenue city in Analytics.")

        fig_monthly = px.line(
            monthly,
            x="month",
            y=["appointments", "completed"],
            markers=True,
            color_discrete_sequence=[BRAND["blue"], BRAND["green"]],
            title="Monthly appointment trends",
        )
        fig_monthly = chart_theme(fig_monthly)
        st.plotly_chart(fig_monthly, use_container_width=True)

    with tab2:
        fig_dept = px.bar(
            dept_perf.sort_values("revenue", ascending=False),
            x="department_name",
            y="revenue",
            color="utilization_rate",
            color_continuous_scale=["#dbeafe", "#22c55e", "#7c3aed"],
            title="Department performance",
        )
        fig_dept = chart_theme(fig_dept)
        st.plotly_chart(fig_dept, use_container_width=True)

        fig_day = px.bar(
            day_of_week,
            x="day_of_week",
            y="appointments",
            color="no_show_rate",
            color_continuous_scale=["#bfdbfe", "#f59e0b", "#e11d48"],
            title="Day-of-week analysis",
        )
        fig_day = chart_theme(fig_day)
        st.plotly_chart(fig_day, use_container_width=True)

    with tab3:
        fig_risk = px.bar(
            risk_by_type.head(12),
            x="department_name",
            y="no_show_rate",
            color="patient_type",
            barmode="group",
            color_discrete_sequence=[BRAND["blue"], BRAND["rose"]],
            title="No-show analysis",
        )
        fig_risk = chart_theme(fig_risk)
        st.plotly_chart(fig_risk, use_container_width=True)
        if not risk_by_type.empty and float(risk_by_type.iloc[0]["no_show_rate"]) > 0.25:
            st.warning(f"⚠ High no-show rate detected in {risk_by_type.iloc[0]['department_name']}.")
def render_manage_records_section() -> None:
    section_label("👤 Manage Records")
    cities, clinics, departments = query_record_reference_data()
    patient_tab, doctor_tab = st.tabs(["Add Patient", "Add Doctor"])

    with patient_tab:
        st.markdown('<div class="form-shell">', unsafe_allow_html=True)
        patient_name = st.text_input("Name", key="manage_patient_name")
        patient_city = st.selectbox("City", cities["city_name"].tolist(), key="manage_patient_city")
        is_returning = st.checkbox("Is Returning", key="manage_patient_returning")
        if st.button("Add Patient", key="manage_patient_submit", use_container_width=True):
            if not patient_name.strip():
                st.error("Patient name is required.")
            else:
                city_id = int(cities.loc[cities["city_name"] == patient_city, "city_id"].iloc[0])
                try:
                    insert_patient_record(patient_name, city_id, is_returning)
                    st.cache_data.clear()
                    st.success("Patient added successfully!")
                except sqlite3.IntegrityError as exc:
                    st.error(f"Could not add patient: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    with doctor_tab:
        st.markdown('<div class="form-shell">', unsafe_allow_html=True)
        doctor_name = st.text_input("Name", key="manage_doctor_name")
        clinic_name = st.selectbox("Clinic", clinics["clinic_name"].tolist(), key="manage_doctor_clinic")
        department_name = st.selectbox(
            "Department",
            departments["department_name"].tolist(),
            key="manage_doctor_department",
        )
        seniority = st.selectbox(
            "Seniority",
            ["Junior", "Mid-level", "Senior"],
            key="manage_doctor_seniority",
        )
        consultation_fee = st.number_input(
            "Consultation Fee",
            min_value=500.0,
            step=100.0,
            key="manage_doctor_fee",
        )
        if st.button("Add Doctor", key="manage_doctor_submit", use_container_width=True):
            if not doctor_name.strip():
                st.error("Doctor name is required.")
            else:
                clinic_id = int(clinics.loc[clinics["clinic_name"] == clinic_name, "clinic_id"].iloc[0])
                department_id = int(
                    departments.loc[departments["department_name"] == department_name, "department_id"].iloc[0]
                )
                try:
                    insert_doctor_record(
                        doctor_name,
                        clinic_id,
                        department_id,
                        seniority,
                        consultation_fee,
                    )
                    st.cache_data.clear()
                    st.success("Doctor added successfully!")
                except sqlite3.IntegrityError as exc:
                    st.error(f"Could not add doctor: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)


def render_pakistan_network_overview_section() -> None:
    section_label("🗺️ Pakistan Network Overview")
    geo_layers = query_pakistan_network_overview()
    cities_df = geo_layers["cities"]
    clinics_df = geo_layers["clinics"]
    doctors_df = geo_layers["doctors"]
    if cities_df.empty:
        st.info("No city-level network data is available yet.")
        return

    m = folium.Map(location=[30.3753, 69.3451], zoom_start=5, tiles="CartoDB positron")

    for clinic in clinics_df.to_dict("records"):
        if clinic["revenue"] < 50000:
            color = "red"
        elif clinic["revenue"] < 100000:
            color = "orange"
        else:
            color = "green"
        folium.CircleMarker(
            location=[clinic["lat"], clinic["lon"]],
            radius=10,
            color=color,
            fill=True,
            fill_opacity=0.7,
            weight=2,
            popup=folium.Popup(
                f"""
                <b>{html.escape(str(clinic['name']))}</b><br>
                City: {html.escape(str(clinic['city']))}<br>
                Revenue: {format_pkr(float(clinic['revenue']))}<br>
                No-show Rate: {float(clinic['no_show_rate']):.1f}%
                """,
                max_width=280,
            ),
            tooltip=f"{clinic['name']} | {clinic['city']}",
        ).add_to(m)

    doctor_cluster = MarkerCluster(name="Doctors").add_to(m)
    for doc in doctors_df.to_dict("records"):
        folium.Marker(
            location=[doc["lat"], doc["lon"]],
            popup=folium.Popup(
                f"""
                <b>Dr {html.escape(str(doc['name']))}</b><br>
                Department: {html.escape(str(doc['department']))}<br>
                No-show Rate: {float(doc['no_show']):.1f}%
                """,
                max_width=260,
            ),
            tooltip=f"Dr {doc['name']}",
            icon=folium.Icon(color="blue", icon="user-md", prefix="fa"),
        ).add_to(doctor_cluster)

    if not clinics_df.empty:
        heat_data = [
            [row["lat"], row["lon"], max(float(row["appointments"]), 1.0)]
            for row in clinics_df.to_dict("records")
        ]
        HeatMap(heat_data, radius=22, blur=18, min_opacity=0.25, name="Demand Heat").add_to(m)

    for city in cities_df.to_dict("records"):
        folium.Circle(
            location=[city["lat"], city["lon"]],
            radius=max(float(city["appointments"]) * 10, 5000),
            color="purple",
            fill=True,
            fill_opacity=0.14,
            weight=2,
            popup=f"{city['city_name']} Performance",
            tooltip=f"{city['city_name']} city performance",
        ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    map_state = st_folium(m, width=1000, height=600, key="pakistan_network_map")

    summary = query_top_performance_summary()
    metric_cards(
        [
            {
                "label": "🏆 Top City by Revenue",
                "value": summary["top_city"],
                "delta": summary["top_city_revenue"],
            },
            {
                "label": "👨‍⚕️ Best Doctor",
                "value": summary["best_doctor"],
                "delta": summary["best_doctor_value"],
            },
            {
                "label": "🧬 Most Active Department",
                "value": summary["top_department"],
                "delta": summary["top_department_value"],
            },
        ]
    )
    if not clinics_df.empty:
        densest_city = clinics_df.groupby("city").size().sort_values(ascending=False).index[0]
        best_revenue_city = cities_df.sort_values("revenue", ascending=False).iloc[0]["city_name"]
        hottest_patient_city = cities_df.sort_values("patients_count", ascending=False).iloc[0]["city_name"]
        st.info(
            f"{densest_city} shows the highest clinic density. {best_revenue_city} leads revenue per network view, while {hottest_patient_city} shows the strongest patient clustering."
        )

    selected_city = None
    if isinstance(map_state, dict):
        popup_text = map_state.get("last_object_clicked_popup")
        if popup_text and popup_text.endswith(" Performance"):
            selected_city = popup_text.replace(" Performance", "")
    if selected_city:
        city_detail = query_city_detail_summary(selected_city)
        if city_detail:
            st.markdown("<br>", unsafe_allow_html=True)
            section_label(f"📍 {selected_city} City Detail")
            metric_cards(
                [
                    {
                        "label": "👨‍⚕️ Total Doctors",
                        "value": city_detail["doctors_count"],
                        "delta": "Clinical supply in the selected city.",
                    },
                    {
                        "label": "🧑‍🤝‍🧑 Total Patients",
                        "value": city_detail["patients_count"],
                        "delta": "Registered patient base currently tied to this city.",
                    },
                    {
                        "label": "💰 Total Revenue",
                        "value": city_detail["revenue"],
                        "delta": "Completed appointment revenue from this city.",
                    },
                    {
                        "label": "🩺 Best Department",
                        "value": city_detail["best_department"],
                        "delta": "Most active department by appointment volume.",
                    },
                ]
            )
            render_auto_insight(
                "🔍 City insight",
                f"{selected_city} stands out through {city_detail['best_department']} and {city_detail['revenue']} in completed-care revenue.",
            )


def add_peak_annotation(fig: go.Figure, x_value, y_value, label: str) -> go.Figure:
    fig.add_annotation(
        x=x_value,
        y=y_value,
        text=label,
        showarrow=True,
        arrowhead=2,
        arrowcolor=BRAND["teal"],
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(15,118,110,0.18)",
    )
    return fig


def build_risk_explanation(patient_type: str, hour: int, lead_days: int, patient_segment: str, chronic_flag: int) -> str:
    reasons = []
    if patient_type == "New":
        reasons.append("new patients are less historically stable")
    if hour in {9, 18}:
        reasons.append("edge appointment slots carry weaker attendance patterns")
    if lead_days >= 10:
        reasons.append("long booking lead time increases uncertainty")
    if patient_segment == "Price Sensitive":
        reasons.append("price-sensitive demand is more attendance-fragile")
    if patient_segment == "Young Professionals":
        reasons.append("young professional appointments are more timing-sensitive")
    if chronic_flag == 1:
        reasons.append("chronic care can slightly stabilize attendance")
    return "High no-show risk due to " + ", ".join(reasons[:3]) + "." if reasons else "Risk is mainly driven by the overall clinic, doctor, and city context."


def takeaway(text: str) -> None:
    st.markdown(
        f"""
        <div class="takeaway">
            <div class="takeaway-title">Final takeaway</div>
            <p class="takeaway-copy">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="footer">
            <h3>MediTrack AI</h3>
            <p>Empowering smarter healthcare decisions through data</p>
            <p>Email: support@meditrack.ai | Phone: +92-300-1234567</p>
            <p>
                <a href="https://linkedin.com" target="_blank">LinkedIn</a>
                <a href="https://github.com" target="_blank">GitHub</a>
            </p>
            <p>© 2026 MediTrack AI. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_overview(city: str, department: str, start_date: str, end_date: str, settings: dict[str, bool]) -> None:
    snapshot = query_snapshot(city, department, start_date, end_date).iloc[0]
    monthly = query_monthly_metrics(city, department, start_date, end_date)
    city_perf = query_city_metrics(city, department, start_date, end_date)
    hero("Network Overview", snapshot)
    section_label("Overview")
    if settings["show_gallery"]:
        render_image_gallery("Network Overview")
    if settings["show_booking"]:
        render_booking_section()
    render_manage_records_section()
    render_pakistan_network_overview_section()
    st.markdown("<br>", unsafe_allow_html=True)

    yoy_growth = 0.0
    if len(monthly) >= 2 and float(monthly["revenue"].iloc[0]) > 0:
        yoy_growth = float(monthly["revenue"].iloc[-1] / monthly["revenue"].iloc[0] - 1)

    metric_cards(
        [
            {
                "label": "💰 Realized Revenue",
                "value": format_pkr(snapshot["realized_revenue"]),
                "delta": "Completed consultations only, showing actual monetized care volume.",
            },
            {
                "label": "✅ Completion Rate",
                "value": format_pct(snapshot["completion_rate"]),
                "delta": "Higher completion means schedule demand is converting into care delivered.",
            },
            {
                "label": "⚠️ No-show Rate",
                "value": format_pct(snapshot["no_show_rate"]),
                "delta": "This is the clearest leakage point for preventable operational loss.",
            },
            {
                "label": "📈 Revenue Growth Arc",
                "value": format_pct(yoy_growth),
                "delta": "Growth matters most when volume and attendance quality move together.",
            },
        ]
    )
    story(PAGE_COPY["Network Overview"]["story"])
    if settings["show_warnings"]:
        render_signal_board(build_overview_warnings(snapshot, monthly, city_perf), "warning")
    if settings["show_recommendations"]:
        render_signal_board(build_overview_recommendations(snapshot, city_perf), "recommendation")

    top_city = city_perf.sort_values("revenue", ascending=False).iloc[0]
    fragile_city = city_perf.sort_values("no_show_rate", ascending=False).iloc[0]
    insight_grid(
        [
            {
                "tag": "Scale leader",
                "title": f"{top_city['city_name']} is the revenue anchor",
                "body": f"It leads the network with {format_pkr(top_city['revenue'])} in realized consultation revenue in the current filtered view.",
            },
            {
                "tag": "Risk pocket",
                "title": f"{fragile_city['city_name']} needs attendance intervention",
                "body": f"It shows the highest no-show rate at {format_pct(fragile_city['no_show_rate'])}, weakening utilization despite existing demand.",
            },
            {
                "tag": "Live data",
                "title": "Booked appointments update the product in real time",
                "body": "New records written through the booking form flow directly into the same SQL-backed analytics layer used throughout the app.",
            },
        ]
    )

    tabs = st.tabs(["Overview Story", "Momentum Signals"])
    with tabs[0]:
        fig_trend = px.line(
            monthly,
            x="month",
            y=["appointments", "completed"],
            markers=True,
            color_discrete_sequence=[BRAND["blue"], BRAND["teal"]],
            title="Network demand trend - completed visits are the metric that truly compounds value",
        )
        fig_trend = chart_theme(fig_trend)
        if not monthly.empty:
            peak_row = monthly.loc[monthly["completed"].idxmax()]
            fig_trend = add_peak_annotation(fig_trend, peak_row["month"], peak_row["completed"], "Peak completed month")
        st.plotly_chart(fig_trend, use_container_width=True)
        if not monthly.empty:
            latest = monthly.iloc[-1]
            render_auto_insight(
                "📈 AI summary",
                f"{int(latest['completed'])} visits were completed in the latest month, showing how much demand converted into delivered care.",
            )
        st.markdown(
            '<p class="chart-note">What is happening: appointment demand trends upward through the filtered period. Why it matters: leaders need to distinguish gross bookings from delivered care. Business impact: stronger completion converts directly into higher clinic efficiency and realized revenue.</p>',
            unsafe_allow_html=True,
        )
    with tabs[1]:
        fig_city = px.scatter(
            city_perf,
            x="appointments",
            y="revenue",
            size="avg_wait",
            color="no_show_rate",
            color_continuous_scale=["#dbeafe", "#38bdf8", "#e11d48"],
            hover_name="city_name",
            title="City performance map - scale, monetization, and attendance risk in one view",
        )
        fig_city = chart_theme(fig_city)
        if not city_perf.empty:
            best_city = city_perf.sort_values("revenue", ascending=False).iloc[0]
            fig_city = add_peak_annotation(fig_city, best_city["appointments"], best_city["revenue"], best_city["city_name"])
        st.plotly_chart(fig_city, use_container_width=True)
        if not city_perf.empty:
            render_auto_insight(
                "🏆 AI summary",
                f"{best_city['city_name']} is the top revenue generating city in the current filtered view.",
            )
        st.markdown(
            '<p class="chart-note">What is happening: cities cluster differently on volume, revenue, and friction. Why it matters: not all large markets are equally healthy. Business impact: investment should favor cities where scale and attendance quality rise together.</p>',
            unsafe_allow_html=True,
        )

    st.success(
        "Live insight: the cleanest growth play is not just more bookings. It is better attendance reliability so booked care becomes delivered, billable care."
    )
    takeaway(PAGE_COPY["Network Overview"]["takeaway"])


def page_utilization(city: str, department: str, start_date: str, end_date: str, settings: dict[str, bool]) -> None:
    snapshot = query_snapshot(city, department, start_date, end_date).iloc[0]
    dept_perf = query_department_metrics(city, department, start_date, end_date)
    clinic_perf = query_clinic_metrics(city, department, start_date, end_date)
    hero("Utilization & Revenue", snapshot)
    section_label("Insights")
    if settings["show_gallery"]:
        render_image_gallery("Utilization & Revenue")

    weakest_department = dept_perf.sort_values("utilization_rate").iloc[0]
    weakest_clinic = clinic_perf.sort_values(["completion_rate", "revenue"], ascending=[True, True]).iloc[0]
    metric_cards(
        [
            {
                "label": "🏥 Department Revenue",
                "value": format_pkr(dept_perf["revenue"].sum()),
                "delta": "Revenue is concentrated in a small number of department and clinic combinations.",
            },
            {
                "label": "📊 Average Utilization",
                "value": format_pct(dept_perf["utilization_rate"].mean()),
                "delta": "A proxy for how well scheduled demand turns into actual consultations.",
            },
            {
                "label": "💳 Avg Consultation Fee",
                "value": format_pkr(dept_perf["avg_fee"].mean()),
                "delta": "Fee mix reflects specialty complexity and doctor seniority composition.",
            },
            {
                "label": "⏱️ Average Wait Time",
                "value": f"{snapshot['avg_wait_time']:.1f} min" if not pd.isna(snapshot["avg_wait_time"]) else "0.0 min",
                "delta": "Long waits usually weaken patient experience and future attendance behavior.",
            },
        ]
    )
    story(PAGE_COPY["Utilization & Revenue"]["story"])
    if settings["show_warnings"]:
        render_signal_board(build_utilization_warnings(snapshot, dept_perf, clinic_perf), "warning")
    if settings["show_recommendations"]:
        render_signal_board(build_utilization_recommendations(dept_perf, clinic_perf), "recommendation")

    insight_grid(
        [
            {
                "tag": "Department risk",
                "title": f"{weakest_department['department_name']} has the weakest schedule quality",
                "body": f"It converts only {format_pct(weakest_department['utilization_rate'])} of bookings into completed consultations, suppressing both efficiency and revenue.",
            },
            {
                "tag": "Clinic risk",
                "title": f"{weakest_clinic['clinic_name']} is the softest clinic in the filtered view",
                "body": f"Completion is at {format_pct(weakest_clinic['completion_rate'])} with {format_pct(weakest_clinic['no_show_rate'])} no-shows, making it a clear intervention candidate.",
            },
            {
                "tag": "Business lens",
                "title": "Premium fees do not save weak execution",
                "body": "Even higher-fee specialties underperform when no-shows and cancellations dilute throughput quality.",
            },
        ]
    )

    tabs = st.tabs(["Department Lens", "Clinic Lens"])
    with tabs[0]:
        fig_dept = px.bar(
            dept_perf.sort_values("revenue", ascending=False),
            x="department_name",
            y="revenue",
            color="utilization_rate",
            color_continuous_scale=["#e0f2fe", "#38bdf8", "#0f766e"],
            title="Department performance - revenue is strongest where utilization stays disciplined",
        )
        fig_dept = chart_theme(fig_dept)
        st.plotly_chart(fig_dept, use_container_width=True)
        render_auto_insight(
            "🧠 AI summary",
            f"{weakest_department['department_name']} needs attention because utilization is only {format_pct(weakest_department['utilization_rate'])}.",
        )
        st.markdown(
            '<p class="chart-note">What is happening: departments vary materially in monetization and schedule conversion. Why it matters: apparent demand can mask operational softness. Business impact: weak-utilization departments should get reminder campaigns, slot redesign, and doctor mix review.</p>',
            unsafe_allow_html=True,
        )
    with tabs[1]:
        fig_clinic = px.scatter(
            clinic_perf,
            x="completion_rate",
            y="revenue",
            size="appointments",
            color="avg_wait",
            hover_name="clinic_name",
            title="Clinic matrix - separate high-efficiency sites from high-friction sites",
            color_continuous_scale=["#dcfce7", "#facc15", "#ef4444"],
        )
        fig_clinic = chart_theme(fig_clinic)
        st.plotly_chart(fig_clinic, use_container_width=True)
        render_auto_insight(
            "🔍 AI summary",
            f"{weakest_clinic['clinic_name']} is the most fragile clinic right now based on conversion and no-show pressure.",
        )
        st.markdown(
            '<p class="chart-note">What is happening: clinics separate into strong and fragile execution clusters. Why it matters: clinic operations determine whether capacity feels premium or overloaded. Business impact: high-performing clinics deserve scale, while weak-conversion clinics need remediation first.</p>',
            unsafe_allow_html=True,
        )

    st.info(
        "Operational insight: MediTrack can improve revenue fastest by tightening conversion where demand already exists. That is usually cheaper and faster than creating entirely new patient demand."
    )
    takeaway(PAGE_COPY["Utilization & Revenue"]["takeaway"])


def page_no_show(city: str, department: str, start_date: str, end_date: str, settings: dict[str, bool]) -> None:
    df = load_detail_data(city, department, start_date, end_date)
    snapshot = query_snapshot(city, department, start_date, end_date).iloc[0]
    risk_by_type = query_no_show_by_type(city, department, start_date, end_date)
    risk_by_hour = query_no_show_by_hour(city, department, start_date, end_date)
    model, features, metrics, importance_df = train_no_show_model(df)
    hero("No-Show Studio", snapshot)
    section_label("Predictions")
    if settings["show_gallery"]:
        render_image_gallery("No-Show Studio")

    riskiest = risk_by_type.iloc[0]
    metric_cards(
        [
            {
                "label": "🤖 Model Accuracy",
                "value": format_pct(metrics["accuracy"]),
                "delta": "The model classifies likely no-shows with strong overall reliability on held-out data.",
            },
            {
                "label": "🎯 F1 Score",
                "value": format_pct(metrics["f1"]),
                "delta": "Useful for balancing false alarms against missed risky appointments.",
            },
            {
                "label": "🧠 ROC AUC",
                "value": format_pct(metrics["roc_auc"]),
                "delta": "Shows how well the model ranks higher-risk appointments ahead of lower-risk ones.",
            },
            {
                "label": "📉 Observed No-show Rate",
                "value": format_pct(snapshot["no_show_rate"]),
                "delta": "This is the baseline attendance risk in the current filtered slice.",
            },
        ]
    )
    story(PAGE_COPY["No-Show Studio"]["story"])
    if settings["show_warnings"]:
        render_signal_board(build_no_show_warnings(snapshot, risk_by_type, risk_by_hour), "warning")
    if settings["show_recommendations"]:
        render_signal_board(build_no_show_recommendations(risk_by_hour, risk_by_type), "recommendation")

    insight_grid(
        [
            {
                "tag": "Highest risk cohort",
                "title": f"{riskiest['patient_type']} patients in {riskiest['department_name']}",
                "body": f"This cohort reaches a no-show rate of {format_pct(riskiest['no_show_rate'])}, making it a prime target for reminder and rescheduling workflows.",
            },
            {
                "tag": "Timing effect",
                "title": "Lead time meaningfully shifts attendance quality",
                "body": "Long booking windows increase schedule uncertainty, especially for new and more timing-sensitive patient groups.",
            },
            {
                "tag": "Operational value",
                "title": "Prediction is only useful when it changes behavior",
                "body": "Use risk scores to trigger outreach, backup waitlists, and more resilient slot construction for high-risk appointments.",
            },
        ]
    )

    tabs = st.tabs(["Risk Signals", "Predictive Copilot"])
    with tabs[0]:
        left, right = st.columns([1.05, 0.95])
        with left:
            fig_type = px.bar(
                risk_by_type.head(10),
                x="department_name",
                y="no_show_rate",
                color="patient_type",
                barmode="group",
                color_discrete_sequence=[BRAND["blue"], BRAND["rose"]],
                title="No-show hotspots - new patients and specialty mix create distinct risk pockets",
            )
            fig_type = chart_theme(fig_type)
            st.plotly_chart(fig_type, use_container_width=True)
            if float(riskiest["no_show_rate"]) > 0.25:
                render_auto_insight(
                    "⚠ AI summary",
                    f"High no-show risk detected for {riskiest['patient_type']} patients in {riskiest['department_name']}.",
                )
        with right:
            fig_importance = px.bar(
                importance_df.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#dbeafe", "#38bdf8", "#0f766e"],
                title="Top model signals driving no-show prediction",
            )
            fig_importance = chart_theme(fig_importance)
            fig_importance.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_importance, use_container_width=True)
            if not importance_df.empty:
                render_auto_insight(
                    "🧠 AI summary",
                    f"{importance_df.iloc[0]['feature'].replace('_', ' ')} is one of the strongest model drivers in the current no-show prediction stack.",
                )
        fig_hour = px.line(
            risk_by_hour,
            x="appointment_hour",
            y="no_show_rate",
            markers=True,
            color_discrete_sequence=[BRAND["amber"]],
            title="Time-of-day effect - edge slots create softer attendance than the network middle",
        )
        fig_hour = chart_theme(fig_hour)
        if not risk_by_hour.empty:
            peak_hour = risk_by_hour.loc[risk_by_hour["no_show_rate"].idxmax()]
            fig_hour = add_peak_annotation(fig_hour, peak_hour["appointment_hour"], peak_hour["no_show_rate"], "Highest risk hour")
        st.plotly_chart(fig_hour, use_container_width=True)
        if not risk_by_hour.empty:
            render_auto_insight(
                "⏰ AI summary",
                f"{int(peak_hour['appointment_hour'])}:00 is the weakest attendance hour, so reminder intensity should be higher there.",
            )
        st.markdown(
            '<p class="chart-note">What is happening: early and late slots carry more no-show pressure than core clinic hours. Why it matters: schedule design itself can reduce waste. Business impact: fragile time bands should receive more aggressive confirmation flows.</p>',
            unsafe_allow_html=True,
        )

    with tabs[1]:
        st.markdown("### Interactive no-show copilot")
        pred_cols = st.columns(4)
        with pred_cols[0]:
            city_value = st.selectbox("City", sorted(df["city_name"].unique()), key="pred_city")
            department_value = st.selectbox("Department", sorted(df["department_name"].unique()), key="pred_department")
            clinic_choices = sorted(df.loc[df["city_name"] == city_value, "clinic_name"].unique())
            clinic_value = st.selectbox("Clinic", clinic_choices, key="pred_clinic")
        with pred_cols[1]:
            doctor_pool = df[
                (df["city_name"] == city_value)
                & (df["clinic_name"] == clinic_value)
                & (df["department_name"] == department_value)
            ]
            if doctor_pool.empty:
                doctor_pool = df[(df["city_name"] == city_value) & (df["clinic_name"] == clinic_value)]
            doctor_value = st.selectbox("Doctor", sorted(doctor_pool["doctor_name"].unique()), key="pred_doctor")
            patient_type = st.selectbox("Patient type", ["New", "Returning"], key="pred_patient_type")
            patient_segment = st.selectbox("Patient segment", sorted(df["patient_segment"].unique()), key="pred_segment")
        with pred_cols[2]:
            channel = st.selectbox("Booking channel", sorted(df["appointment_channel"].unique()), key="pred_channel")
            weekday = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], key="pred_weekday")
            hour = st.slider("Appointment hour", 9, 18, 15, key="pred_hour")
        with pred_cols[3]:
            lead_days = st.slider("Lead days", 0, 21, 5, key="pred_lead")
            chronic_flag = st.checkbox("Chronic care patient", value=False, key="pred_chronic")

        doctor_row = (
            df[df["doctor_name"] == doctor_value][
                ["seniority_level", "years_experience", "popularity_score", "consultation_fee", "traffic_index"]
            ]
            .drop_duplicates()
            .iloc[0]
        )
        input_frame = pd.DataFrame(
            [
                {
                    "city_name": city_value,
                    "clinic_name": clinic_value,
                    "department_name": department_value,
                    "seniority_level": doctor_row["seniority_level"],
                    "patient_type": patient_type,
                    "appointment_channel": channel,
                    "patient_segment": patient_segment,
                    "appointment_hour": hour,
                    "lead_days": lead_days,
                    "consultation_fee": float(doctor_row["consultation_fee"]),
                    "is_peak_slot": int(hour in {10, 11, 15, 16}),
                    "years_experience": int(doctor_row["years_experience"]),
                    "popularity_score": float(doctor_row["popularity_score"]),
                    "traffic_index": float(doctor_row["traffic_index"]),
                    "chronic_flag": int(chronic_flag),
                    "day_of_week": weekday,
                }
            ]
        )[features]
        risk_probability = float(model.predict_proba(input_frame)[0, 1])
        explanation = build_risk_explanation(patient_type, hour, lead_days, patient_segment, int(chronic_flag))

        left, right = st.columns([0.9, 1.1])
        with left:
            st.metric("Predicted no-show probability", f"{risk_probability * 100:.1f}%")
            st.markdown(
                f"**Recommended action:** {'Standard reminder' if risk_probability < 0.18 else 'Reminder plus confirmation' if risk_probability < 0.30 else 'High-touch reminder, waitlist backup, or slot redesign'}"
            )
            st.markdown(explanation)
        with right:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_probability * 100,
                    number={"suffix": "%"},
                    title={"text": "Predicted no-show risk"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": BRAND["teal"]},
                        "steps": [
                            {"range": [0, 18], "color": "#dcfce7"},
                            {"range": [18, 30], "color": "#fef3c7"},
                            {"range": [30, 100], "color": "#fee2e2"},
                        ],
                    },
                )
            )
            gauge = chart_theme(gauge)
            st.plotly_chart(gauge, use_container_width=True)

        if risk_probability < 0.18:
            st.success(f"Predicted no-show likelihood: {risk_probability * 100:.1f}%. This appointment looks comparatively stable.")
        elif risk_probability < 0.30:
            st.warning(f"Predicted no-show likelihood: {risk_probability * 100:.1f}%. A reminder plus confirmation step is recommended.")
        else:
            st.error(f"Predicted no-show likelihood: {risk_probability * 100:.1f}%. This appointment deserves high-touch intervention.")

    takeaway(PAGE_COPY["No-Show Studio"]["takeaway"])


def page_doctor(city: str, department: str, start_date: str, end_date: str, settings: dict[str, bool]) -> None:
    snapshot = query_snapshot(city, department, start_date, end_date).iloc[0]
    doctor_perf = query_doctor_metrics(city, department, start_date, end_date)
    hero("Doctor Explorer", snapshot)
    section_label("Care Team")
    if settings["show_gallery"]:
        render_image_gallery("Doctor Explorer")

    doctor_name = st.selectbox("Choose a doctor", doctor_perf["doctor_name"].tolist())
    selected = doctor_perf[doctor_perf["doctor_name"] == doctor_name].iloc[0]
    doctor_monthly = query_doctor_monthly(doctor_name, city, department, start_date, end_date)
    peer_set = doctor_perf[doctor_perf["department_name"] == selected["department_name"]].copy()
    top_peer = peer_set.sort_values("revenue", ascending=False).iloc[0]

    metric_cards(
        [
            {
                "label": "👨‍⚕️ Doctor Revenue",
                "value": format_pkr(selected["revenue"]),
                "delta": "Completed consultation revenue credited to the selected clinician.",
            },
            {
                "label": "📌 Completion Rate",
                "value": format_pct(selected["completion_rate"]),
                "delta": "How effectively booked slots convert into delivered care.",
            },
            {
                "label": "🚨 No-show Exposure",
                "value": format_pct(selected["no_show_rate"]),
                "delta": "Higher no-show exposure often reflects patient mix, timing, and demand positioning.",
            },
            {
                "label": "⭐ Patient Satisfaction",
                "value": f"{selected['avg_satisfaction']:.2f}/5" if not pd.isna(selected["avg_satisfaction"]) else "n/a",
                "delta": "Experience quality strengthens retention and future attendance behavior.",
            },
        ]
    )
    story(PAGE_COPY["Doctor Explorer"]["story"])
    if settings["show_warnings"]:
        render_signal_board(build_doctor_warnings(selected, peer_set), "warning")
    if settings["show_recommendations"]:
        render_signal_board(build_doctor_recommendations(selected, top_peer), "recommendation")
    insight_grid(
        [
            {
                "tag": "Profile",
                "title": f"{selected['doctor_name']} works in {selected['department_name']}",
                "body": f"The selected doctor operates in {selected['city_name']} as a {selected['seniority_level']} clinician with {selected['appointments']:,} appointments in scope.",
            },
            {
                "tag": "Benchmark",
                "title": f"{top_peer['doctor_name']} is the department benchmark",
                "body": f"Benchmark revenue in this department currently sits at {format_pkr(top_peer['revenue'])}, creating a practical coaching reference.",
            },
            {
                "tag": "Retention quality",
                "title": "Experience quality shapes future demand quality",
                "body": "Clinicians with stronger satisfaction and cleaner completion patterns create healthier schedules over time.",
            },
        ]
    )

    tabs = st.tabs(["Doctor Story", "Peer Comparison"])
    with tabs[0]:
        fig_doc_trend = px.line(
            doctor_monthly,
            x="month",
            y=["revenue", "appointments"],
            markers=True,
            color_discrete_sequence=[BRAND["blue"], BRAND["teal"]],
            title="Doctor trajectory - revenue and schedule demand over time",
        )
        fig_doc_trend = chart_theme(fig_doc_trend)
        st.plotly_chart(fig_doc_trend, use_container_width=True)
        render_auto_insight(
            "👨‍⚕️ AI summary",
            f"{selected['doctor_name']} currently converts {format_pct(selected['completion_rate'])} of scheduled demand into completed care.",
        )
    with tabs[1]:
        fig_peer = px.bar(
            peer_set.sort_values("revenue", ascending=False).head(12),
            x="doctor_name",
            y="revenue",
            color="no_show_rate",
            color_continuous_scale=["#dcfce7", "#60a5fa", "#f43f5e"],
            title=f"{selected['department_name']} comparison - strong doctors combine revenue with cleaner attendance",
        )
        fig_peer = chart_theme(fig_peer)
        fig_peer.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_peer, use_container_width=True)
        render_auto_insight(
            "🏅 AI summary",
            f"{top_peer['doctor_name']} is the strongest peer benchmark in {selected['department_name']} by realized revenue.",
        )
        st.markdown(
            '<p class="chart-note">What is happening: peer doctors differ in realized revenue and no-show exposure. Why it matters: coaching should be individualized instead of department-wide. Business impact: leadership can protect top performers and intervene where demand quality is soft.</p>',
            unsafe_allow_html=True,
        )

    st.info(
        "Doctor insight: the best-performing clinicians are not just busy. They also attract steadier attendance and stronger patient experience, which protects long-term capacity quality."
    )
    takeaway(PAGE_COPY["Doctor Explorer"]["takeaway"])


def main() -> None:
    inject_styles()
    render_top_brand()
    if not ensure_database():
        st.error(f"Database not found at `{DB_PATH}`. Run `python quickbite/meditrackdata.py` first.")
        return

    ensure_appointments_support_scheduled()
    page, city, department, date_range, settings = render_sidebar()
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.warning("Please choose a valid date range.")
        return
    start_date, end_date = date_range

    snapshot = query_snapshot(city, department, str(start_date), str(end_date)).iloc[0]
    if int(snapshot["appointments"]) == 0:
        st.warning("The current filters returned no data. Adjust the city, department, or date range to continue.")
        render_footer()
        return

    if page == "Dashboard Overview":
        saas_dashboard_overview(city, department, str(start_date), str(end_date), settings)
    elif page == "Doctors":
        saas_doctors_page(city, department, str(start_date), str(end_date), settings)
    elif page == "Patients":
        saas_patients_page(city, department, str(start_date), str(end_date), settings)
    elif page == "Book Appointment":
        saas_book_appointment_page(settings)
    else:
        saas_analytics_page(city, department, str(start_date), str(end_date), settings)

    render_floating_chatbot(page)
    render_footer()


if __name__ == "__main__":
    main()
