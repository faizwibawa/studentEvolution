import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px

# --- Configuration & Setup ---
st.set_page_config(page_title="College Strategy ABM", layout="wide")

# --- Agent Definition ---
class StudentAgent:
    def __init__(self, template_name, templates_config, actions_config, events_config, mutation_rate):
        self.template = template_name
        self.templates_config = templates_config
        self.probs = templates_config[template_name]
        self.actions_config = actions_config
        self.events_config = events_config
        self.mutation_rate = mutation_rate
        self.all_templates = list(templates_config.keys())
        self.action_names = list(actions_config.keys())
        
        self.event_streaks = {event: 0 for event in events_config.keys()}
        self.recovery_action = max(self.action_names, key=lambda k: self.actions_config[k]["energy"])
        
        self.energy = random.randint(50, 100)
        self.stress = random.randint(0, 30)
        self.score = random.randint(0, 20)
        self.social = random.randint(10, 50) 
        
    def step(self):
        # 0. Mutation Phase
        if random.random() < self.mutation_rate:
            possible_mutations = [t for t in self.all_templates if t != self.template]
            if possible_mutations:
                self.template = random.choice(possible_mutations)
                self.probs = self.templates_config[self.template]

        # 1. Take Action Phase
        if self.energy <= 10:
            action = self.recovery_action
        else:
            probabilities = [self.probs.get(a, 0) for a in self.action_names]
            action = np.random.choice(self.action_names, p=probabilities)
            
        effect = self.actions_config[action]
        self.apply_stat_changes(effect["energy"], effect["stress"], effect["score"], effect["social"])
        
        # 2. Check and Trigger Events Phase
        self.check_events()

    def apply_stat_changes(self, energy_change, stress_change, score_change, social_change):
        self.energy = max(0, min(100, self.energy + energy_change))
        self.stress = max(0, min(100, self.stress + stress_change))
        self.score = max(0, min(100, self.score + score_change))
        self.social = max(0, min(100, self.social + social_change))

    def check_events(self):
        for event_name, ev_data in self.events_config.items():
            current_stat_val = getattr(self, ev_data["stat"])
            
            condition_met = False
            if ev_data["condition"] == ">" and current_stat_val > ev_data["threshold"]:
                condition_met = True
            elif ev_data["condition"] == "<" and current_stat_val < ev_data["threshold"]:
                condition_met = True
                
            if condition_met:
                self.event_streaks[event_name] += 1
                if self.event_streaks[event_name] >= ev_data["days_req"]:
                    if random.random() < ev_data["chance"]:
                        self.apply_stat_changes(ev_data["energy_eff"], ev_data["stress_eff"], ev_data["score_eff"], ev_data["social_eff"])
                        self.event_streaks[event_name] = 0 
            else:
                self.event_streaks[event_name] = 0

    def get_fitness(self, weights):
        base_fitness = (self.score * weights["score"]) + (self.energy * weights["energy"]) + (self.social * weights["social"])
        stress_penalty = self.stress * weights["stress_penalty"]
        fitness = base_fitness - stress_penalty
        return max(1, fitness)

# --- Simulation Logic ---
def run_simulation(num_students, num_semesters, days_per_semester, templates_config, actions_config, events_config, fitness_weights, mutation_rate):
    templates = list(templates_config.keys())
    population = [StudentAgent(random.choice(templates), templates_config, actions_config, events_config, mutation_rate) for _ in range(num_students)]
    history = []

    for semester in range(num_semesters):
        for day in range(days_per_semester):
            for agent in population:
                agent.step()
                
        counts = {t: 0 for t in templates}
        for agent in population:
            counts[agent.template] += 1
        
        record = {"Semester": semester + 1}
        record.update(counts)
        history.append(record)
        
        template_fitness = {t: 0 for t in templates}
        for agent in population:
            template_fitness[agent.template] += agent.get_fitness(fitness_weights)
            
        total_fitness = sum(template_fitness.values())
        
        if total_fitness > 0:
            weights = [template_fitness[t] / total_fitness for t in templates]
        else:
            weights = [1.0 / len(templates) for _ in templates]
            
        population = [StudentAgent(np.random.choice(templates, p=weights), templates_config, actions_config, events_config, mutation_rate) for _ in range(num_students)]

    return pd.DataFrame(history)

# --- Streamlit UI ---
st.title("College Student Strategies: Evolutionary ABM")
st.markdown("A sandbox to test how student archetypes survive when balancing grades, energy, stress, and social life.")

# --- UI: Sidebar Parameters ---
st.sidebar.header("Simulation Time")
num_students = st.sidebar.slider("Students per Semester", 50, 500, 200, 50)
num_semesters = st.sidebar.slider("Number of Semesters", 5, 50, 20, 1)
days_per_semester = st.sidebar.slider("Days in a Semester", 10, 100, 30, 5)

st.sidebar.header("Evolutionary Pressures")
score_weight = st.sidebar.slider("Score Importance", 0.0, 3.0, 1.0, 0.1)
social_weight = st.sidebar.slider("Social Importance", 0.0, 3.0, 0.5, 0.1)
energy_weight = st.sidebar.slider("Energy Importance", 0.0, 3.0, 0.0, 0.1)
stress_penalty = st.sidebar.slider("Stress Penalty", 0.0, 3.0, 0.5, 0.1)

fitness_weights = {
    "score": score_weight, 
    "social": social_weight, 
    "energy": energy_weight, 
    "stress_penalty": stress_penalty
}

st.sidebar.header("Genetics")
mutation_rate = st.sidebar.slider("Daily Mutation Chance", 0.0, 0.1, 0.01, 0.001, format="%.3f")

# --- UI: Action Customization ---
st.subheader("1. Actions")
default_actions_df = pd.DataFrame({
    "Action Name": ["Study", "Rest", "Party", "Part-Time Job"],
    "Energy Effect": [-15, 25, -10, -20],
    "Stress Effect": [10, -20, -30, 20],
    "Score Effect": [10, 0, -5, 0],
    "Social Effect": [-10, 0, 30, 10]
})
edited_actions_df = st.data_editor(default_actions_df, num_rows="fixed", width="stretch")

actions_config = {}
for _, row in edited_actions_df.iterrows():
    name = str(row["Action Name"]).strip()
    if name:
        actions_config[name] = {
            "energy": float(row["Energy Effect"]), 
            "stress": float(row["Stress Effect"]), 
            "score": float(row["Score Effect"]),
            "social": float(row["Social Effect"])
        }
action_names = list(actions_config.keys())

# --- UI: Template Customization ---
st.subheader("2. Templates")
default_templates_df = pd.DataFrame({
    "Template Name": ["Grinder", "Slacker", "Balanced", "Socialite"],
    action_names[0]: [0.7, 0.1, 0.3, 0.1],
    action_names[1]: [0.2, 0.4, 0.3, 0.2],
    action_names[2]: [0.1, 0.5, 0.2, 0.6],
    action_names[3]: [0.0, 0.0, 0.2, 0.1]
})
edited_templates_df = st.data_editor(default_templates_df, num_rows="dynamic", width="stretch")

# --- UI: Random Events Customization ---
st.subheader("3. Conditional Random Events")
default_events_df = pd.DataFrame({
    "Event Name": ["Burnout", "Panic Monster", "Sickness", "FOMO"],
    "Target Stat": ["stress", "score", "energy", "social"],
    "Condition": [">", "<", "<", "<"],
    "Threshold": [80, 20, 30, 15],
    "Days Required": [4, 7, 3, 5],
    "Probability": [0.6, 0.8, 0.3, 0.7], 
    "Energy Hit": [-40, 10, -30, -10],
    "Stress Hit": [0, 40, 10, 30],
    "Score Hit": [-10, 20, -10, -5],
    "Social Hit": [-20, -10, -20, 0]
})

edited_events_df = st.data_editor(
    default_events_df, 
    num_rows="dynamic",
    column_config={
        "Target Stat": st.column_config.SelectboxColumn("Target Stat", options=["energy", "stress", "score", "social"], required=True),
        "Condition": st.column_config.SelectboxColumn("Condition", options=[">", "<"], required=True),
        "Probability": st.column_config.NumberColumn("Probability", min_value=0.0, max_value=1.0, required=True)
    },
    width="stretch"
)

if st.button("Run Evolutionary Simulation", type="primary"):
    if len(action_names) != 4:
        st.error("Please ensure exactly 4 unique action names are defined.")
    else:
        with st.spinner("Simulating generations..."):
            
            templates_config = {}
            for _, row in edited_templates_df.iterrows():
                name = str(row["Template Name"]).strip()
                if name: 
                    raw_probs = np.array([float(row[act]) for act in action_names])
                    raw_probs = np.nan_to_num(raw_probs, 0)
                    normalized_probs = raw_probs / sum(raw_probs) if sum(raw_probs) > 0 else [0.25]*4
                    templates_config[name] = {action_names[i]: normalized_probs[i] for i in range(4)}

            events_config = {}
            for _, row in edited_events_df.iterrows():
                name = str(row["Event Name"]).strip()
                if name:
                    events_config[name] = {
                        "stat": row["Target Stat"],
                        "condition": row["Condition"],
                        "threshold": float(row["Threshold"]),
                        "days_req": int(row["Days Required"]),
                        "chance": float(row["Probability"]),
                        "energy_eff": float(row["Energy Hit"]),
                        "stress_eff": float(row["Stress Hit"]),
                        "score_eff": float(row["Score Hit"]),
                        "social_eff": float(row["Social Hit"])
                    }
                    
            df_history = run_simulation(num_students, num_semesters, days_per_semester, templates_config, actions_config, events_config, fitness_weights, mutation_rate)
            
        st.subheader("Template Evolution Over Time")
        df_melted = df_history.melt(id_vars=["Semester"], var_name="Template", value_name="Population")
        
        fig = px.area(
            df_melted, 
            x="Semester", 
            y="Population", 
            color="Template",
            title="Student Population Strategy Distribution",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        st.plotly_chart(fig, width="stretch")

        with st.expander("View Raw Data & Final States"):
            st.dataframe(df_history.set_index("Semester"), width="stretch")