
import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

# ==============================
# 1. ENVIRONMENT AND DATABASE
# ==============================

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY .env file mein set nahi hai."
    )

DB_NAME = "digital_marketing.db"


def get_db():
    return sqlite3.connect(DB_NAME)


def initialize_database():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_name TEXT NOT NULL,
                business_name TEXT NOT NULL,
                platform TEXT NOT NULL,
                goal TEXT NOT NULL,
                audience TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                contact TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'New',
                notes TEXT,
                created_at TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER NOT NULL,
                impressions INTEGER NOT NULL,
                clicks INTEGER NOT NULL,
                conversions INTEGER NOT NULL,
                spend REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (campaign_id)
                    REFERENCES campaigns(id)
            )
        """)


# ==============================
# 2. CAMPAIGN TOOLS
# ==============================

@tool
def save_campaign(
    campaign_name: str,
    business_name: str,
    platform: str,
    goal: str,
    audience: str,
    content: str
) -> str:
    """Save a marketing campaign and its generated content to the database.
    Use only when the user asks to save a campaign."""

    if not all([
        campaign_name.strip(),
        business_name.strip(),
        platform.strip(),
        goal.strip(),
        audience.strip(),
        content.strip()
    ]):
        return "Error: Campaign ki tamam details zaroori hain."

    with get_db() as db:
        cursor = db.execute("""
            INSERT INTO campaigns
            (campaign_name, business_name, platform,
             goal, audience, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            campaign_name, business_name, platform,
            goal, audience, content,
            datetime.now().isoformat(timespec="seconds")
        ))

        campaign_id = cursor.lastrowid

    return (
        f"Campaign successfully saved. "
        f"Campaign ID: {campaign_id}. "
        f"Name: {campaign_name}"
    )


@tool
def list_campaigns() -> str:
    """Show saved marketing campaigns and their IDs."""

    with get_db() as db:
        rows = db.execute("""
            SELECT id, campaign_name, business_name,
                   platform, goal, created_at
            FROM campaigns
            ORDER BY id DESC
        """).fetchall()

    if not rows:
        return "Abhi database mein koi campaign saved nahi hai."

    return "\n".join(
        f"ID: {r[0]} | Campaign: {r[1]} | "
        f"Business: {r[2]} | Platform: {r[3]} | "
        f"Goal: {r[4]} | Created: {r[5]}"
        for r in rows
    )


# ==============================
# 3. LEAD MANAGEMENT TOOLS
# ==============================

@tool
def add_lead(
    name: str,
    contact: str,
    source: str,
    notes: str = ""
) -> str:
    """Save a potential customer's name, contact, source and notes."""

    if not name.strip() or not contact.strip():
        return "Error: Name aur contact zaroori hain."

    with get_db() as db:
        cursor = db.execute("""
            INSERT INTO leads
            (name, contact, source, status, notes, created_at)
            VALUES (?, ?, ?, 'New', ?, ?)
        """, (
            name, contact, source, notes,
            datetime.now().isoformat(timespec="seconds")
        ))

        lead_id = cursor.lastrowid

    return f"Lead saved successfully. Lead ID: {lead_id}, Name: {name}"


@tool
def search_leads(query: str) -> str:
    """Search saved leads by name or contact information."""

    term = f"%{query.strip()}%"

    with get_db() as db:
        rows = db.execute("""
            SELECT id, name, contact, source, status, notes
            FROM leads
            WHERE name LIKE ? OR contact LIKE ?
            ORDER BY id DESC
        """, (term, term)).fetchall()

    if not rows:
        return "Is search ke liye koi lead nahi mili."

    return "\n".join(
        f"ID: {r[0]} | Name: {r[1]} | Contact: {r[2]} | "
        f"Source: {r[3]} | Status: {r[4]} | Notes: {r[5]}"
        for r in rows
    )


@tool
def update_lead_status(contact: str, new_status: str) -> str:
    """Update a lead status using its exact contact information.
    Allowed statuses: New, Contacted, Qualified, Converted, Lost."""

    allowed = {
        "new": "New",
        "contacted": "Contacted",
        "qualified": "Qualified",
        "converted": "Converted",
        "lost": "Lost"
    }

    status = allowed.get(new_status.strip().lower())

    if not status:
        return "Invalid status. Use New, Contacted, Qualified, Converted or Lost."

    with get_db() as db:
        cursor = db.execute("""
            UPDATE leads SET status = ?
            WHERE contact = ?
        """, (status, contact.strip()))

        changed = cursor.rowcount

    if changed == 0:
        return "Is exact contact ke saath koi lead nahi mili."

    return f"{changed} lead record(s) updated to {status}."


# ==============================
# 4. CAMPAIGN ANALYTICS TOOLS
# ==============================

@tool
def add_campaign_metrics(
    campaign_id: int,
    impressions: int,
    clicks: int,
    conversions: int,
    spend: float
) -> str:
    """Save actual campaign performance numbers for an existing campaign.
    All counts and advertising spend must be zero or greater."""

    if min(impressions, clicks, conversions, spend) < 0:
        return "Error: Metrics negative nahi ho sakte."

    if clicks > impressions:
        return "Error: Clicks impressions se zyada nahi ho sakte."

    if conversions > clicks:
        return "Error: Conversions clicks se zyada nahi ho sakte."

    with get_db() as db:
        campaign = db.execute(
            "SELECT id FROM campaigns WHERE id = ?",
            (campaign_id,)
        ).fetchone()

        if not campaign:
            return f"Campaign ID {campaign_id} database mein nahi mili."

        cursor = db.execute("""
            INSERT INTO metrics
            (campaign_id, impressions, clicks, conversions,
             spend, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            campaign_id, impressions, clicks, conversions,
            spend, datetime.now().isoformat(timespec="seconds")
        ))

        metric_id = cursor.lastrowid

    return (
        f"Metrics saved successfully. Metric ID: {metric_id}, "
        f"Campaign ID: {campaign_id}"
    )


@tool
def campaign_report(campaign_id: int) -> str:
    """Calculate a performance report using actual saved campaign metrics."""

    with get_db() as db:
        campaign = db.execute("""
            SELECT campaign_name FROM campaigns WHERE id = ?
        """, (campaign_id,)).fetchone()

        if not campaign:
            return f"Campaign ID {campaign_id} database mein nahi mili."

        row = db.execute("""
            SELECT SUM(impressions), SUM(clicks),
                   SUM(conversions), SUM(spend)
            FROM metrics WHERE campaign_id = ?
        """, (campaign_id,)).fetchone()

    impressions = row[0] or 0
    clicks = row[1] or 0
    conversions = row[2] or 0
    spend = row[3] or 0.0

    ctr = (clicks / impressions * 100) if impressions else 0
    conversion_rate = (
        conversions / clicks * 100
    ) if clicks else 0
    cost_per_conversion = (
        spend / conversions
    ) if conversions else None

    cost_text = (
        f"Rs. {cost_per_conversion:.2f}"
        if cost_per_conversion is not None
        else "N/A (no conversions recorded)"
    )

    return (
        f"Campaign: {campaign[0]}\n"
        f"Impressions: {impressions}\n"
        f"Clicks: {clicks}\n"
        f"Conversions: {conversions}\n"
        f"Ad spend: Rs. {spend:.2f}\n"
        f"CTR: {ctr:.2f}%\n"
        f"Conversion rate: {conversion_rate:.2f}%\n"
        f"Cost per conversion: {cost_text}\n"
        "Report calculated from saved data."
    )


# ==============================
# 5. AI AGENT
# ==============================

initialize_database()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

tools = [
    save_campaign,
    list_campaigns,
    add_lead,
    search_leads,
    update_lead_status,
    add_campaign_metrics,
    campaign_report
]

system_prompt = """
You are an Advanced Digital Marketing AI Agent.

Your responsibilities:
1. Create Facebook, Instagram and LinkedIn posts.
2. Generate engaging captions, hashtags and calls to action.
3. Write advertising copy for products and services.
4. Create 7-day and 30-day content calendars.
5. Suggest marketing strategies and target audiences.
6. Draft email marketing messages and promotions.
7. Save campaigns when the user explicitly asks.
8. Manage marketing leads using the available tools.
9. Record and analyze campaign metrics using actual supplied data.

Rules:
- Help the user in English, Urdu or Roman Urdu as requested.
- Ask for missing business details when necessary.
- Do not invent campaign performance, sales, clicks or conversions.
- Use database tools when the user asks to save, search or update records.
- Never say a campaign or lead was saved unless the tool confirms it.
- Never claim that ads were published or emails sent. These tools only
  generate drafts and manage local records.
- Do not claim live social media account access.
- Explain marketing calculations clearly.
- Give practical, professional, beginner-friendly answers.
"""

memory = MemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=memory
)

config = {
    "configurable": {
        "thread_id": "digital_marketing_demo_001"
    }
}


def extract_answer(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)

    return str(content)


# ==============================
# 6. TERMINAL CHAT
# ==============================

print("=" * 48)
print("   ADVANCED DIGITAL MARKETING AI AGENT")
print("=" * 48)
print("Content | Ads | Calendar | Campaigns | Leads | Analytics")
print("Type 'exit' to stop.\n")

while True:
    try:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Agent: Allah Hafiz! Good luck with your marketing.")
            break

        if not user_input:
            continue

        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            },
            config=config
        )

        answer = extract_answer(result["messages"][-1].content)
        print(f"\nDigital Marketing Agent: {answer}\n")

    except KeyboardInterrupt:
        print("\nAgent stopped.")
        break

    except Exception as error:
        print(f"\nError: {error}")
        print(
            "API key, model name aur installed packages check karein.\n"
        )


# Pakistan mein ek clothing brand ke liye 7 din ka social media content calendar banao. Har din ka post idea, Roman Urdu caption, hashtags aur CTA do.
# 
# Mere digital marketing database mein Haris Clothing Store ke liye ek campaign banao. Platform Instagram, goal sales increase karna, target audience Pakistan ke 18 se 35 saal ke log hon. Campaign ka naam, caption aur tamam details database mein save karo. Save karne ke baad confirmation do.
# 
# Ek potential customer ki lead database mein save karo. Name: Ali Khan, contact: 03001234567, source: Instagram, status: New, notes: Gym membership mein interested. Save karne ke baad lead ko database se search karke verify bhi karo.
# 
# Haris Fitness Gym ki saved campaigns ki report banao. Campaign names, platforms aur goals batao. Agar impressions, clicks, ad spend ya conversions ka data database mein nahi hai to clearly batao. Koi numbers khud se mat banana.
# 
# Database mein Ali Khan ki lead search karo. Agar woh maujood hai to uska status Contacted kar do aur dobara search karke confirm karo ke status update hua hai. Agar lead nahi milti to naya record khud se mat banana.
# 
# Mera project **Advanced Digital Marketing AI Agent** hai.

# Ye AI agent businesses ki digital marketing mein help karta hai.

# 1. Facebook aur Instagram ke liye marketing content, captions aur hashtags banata hai.
# 2. 7-day aur 30-day social media content calendars banata hai.
# 3. Marketing campaigns ke ideas aur advertising copy generate karta hai.
# 4. Marketing campaigns ko database mein save karta hai aur baad mein search bhi kar sakta hai.
# 5. Potential customers ki leads save karta hai, unhein search karta hai aur unka status update karta hai.
# 6. Campaign ki performance track karta hai, jaise impressions, clicks, conversions aur advertising cost.
# 7. Marketing reports aur email drafts banane mein help karta hai.

# Is project mein Python, Gemini AI, LangChain, LangGraph aur SQLite database use kiye gaye hain.

# Simple words mein:
# Ye ek AI marketing assistant hai jo marketing content banane, campaigns manage karne, customers ki leads save karne aur marketing performance track karne mein help karta hai.
# #
