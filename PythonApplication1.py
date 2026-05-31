import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px

# --- Configuration & Setup ---
st.set_page_config(page_title="College Strategy ABM", layout="wide")

# --- Agent Definition ---
class StudentAgent:
    def __init__(self, template_name, templates_config, actions_config):
        self.template = template_name
        self.probs = templates_config[template_name]
        self.actions_config = actions_config
        self.action_names = list(actions_config.keys())
        
        # Identify the action that restores the most energy for exhaustion fallback
        self.recovery_action = max(self.action_names, key=lambda k: self.actions_config[k]["energy"])
        
        # Initial randomized fixed traits
        self.energy = random.randint(50, 100)
        self.stress = random.randint(0, 30)
        self.score = random.randint(0, 20)
        
    def step(self):
        # If exhausted, forced to take the highest energy-yielding action
        if self.energy <= 10:
            action = self.recovery_action
        else:
            # Choose action based on template probabilities
            probabilities = [self.probs.get(a, 0) for a in self.action_names]
            action = np.random.choice(self.action_names, p=probabilities)
            
        # Apply action effects
        effect = self.actions_config[action]
        self.energy = max(0, min(100, self.energy + effect["energy"]))
        self.stress = max(0, min(100, self.stress + effect["stress"]))
        self.score = max(0, min(100, self.score + effect["score"]))

    def get_fitness(self, weights):
        # Calculate fitness based on user-defined evolutionary pressures
        base_fitness = (self.score * weights["score"]) + (self.energy * weights["energy"])
        stress_penalty = self.stress * weights["stress_penalty"]
        
        fitness = base_fitness - stress_penalty
        return max(1, fitness) # Ensure minimum fitness of 1 to stay in the gene pool

# --- Simulation Logic ---
def run_simulation(num_students, num_semesters, days_per_semester, templates_config, actions_config, fitness_weights):
    templates = list(templates_config.keys())
    
    # Initialize first generation evenly
    population = [StudentAgent(random.choice(templates), templates_config, actions_config) for _ in range(num_students)]
    history = []

    for semester in range(num_semesters):
        # Run the semester
        for day in range(days_per_semester):
            for agent in population:
                agent.step()
                
        # Record population distribution
        counts = {t: 0 for t in templates}
        for agent in population:
            counts[agent.template] += 1
        
        record = {"Semester": semester + 1}
        record.update(counts)
        history.append(record)
        
        # --- EVOLUTION (Inheritance) ---
        template_fitness = {t: 0 for t in templates}
        for agent in population:
            template_fitness[agent.template] += agent.get_fitness(fitness_weights)
            
        total_fitness = sum(template_fitness.values())
        
        # Create next generation based on fitness weights
        if total_fitness > 0:
            weights = [template_fitness[t] / total_fitness for t in templates]
        else:
            weights = [1.0 / len(templates) for _ in templates] # Fallback
            
        population = [StudentAgent(np.random.choice(templates, p=weights), templates_config, actions_config) for _ in range(num_students)]

    return pd.DataFrame(history)

# --- Streamlit UI ---
st.title("🎓 College Student Strategies: Evolutionary ABM")
st.markdown("Customize actions, define templates, and set the evolutionary pressures (fitness) that determine which strategies survive.")

# --- UI: Sidebar Parameters ---
st.sidebar.header("⏱️ Simulation Time")
num_students = st.sidebar.slider("Students per Semester", 50, 500, 200, 50)
num_semesters = st.sidebar.slider("Number of Semesters", 5, 50, 20, 1)
days_per_semester = st.sidebar.slider("Days in a Semester", 10, 100, 30, 5)

st.sidebar.header("🧬 Evolutionary Pressures")
st.sidebar.markdown("What determines a student's 'fitness' to pass on their strategy?")
score_weight = st.sidebar.slider("Score Importance", 0.0, 3.0, 1.0, 0.1)
energy_weight = st.sidebar.slider("Energy Importance", 0.0, 3.0, 0.0, 0.1)
stress_penalty = st.sidebar.slider("Stress Penalty", 0.0, 3.0, 0.5, 0.1)

fitness_weights = {
    "score": score_weight,
    "energy": energy_weight,
    "stress_penalty": stress_penalty
}

# --- UI: Action Customization ---
st.subheader("⚡ Customize 4 Actions")
st.markdown("Define the 4 actions available to students and how they affect their fixed traits.")

default_actions_df = pd.DataFrame({
    "Action Name": ["Study", "Rest", "Party", "Part-Time Job"],
    "Energy Effect": [-15, 25, -10, -20],
    "Stress Effect": [10, -20, -30, 20],
    "Score Effect": [10, 0, -5, 0]
})

edited_actions_df = st.data_editor(default_actions_df, num_rows="fixed", use_container_width=True)

# Parse actions to ensure they are available for the templates editor
actions_config = {}
for _, row in edited_actions_df.iterrows():
    name = str(row["Action Name"]).strip()
    if name:
        actions_config[name] = {
            "energy": float(row["Energy Effect"]),
            "stress": float(row["Stress Effect"]),
            "score": float(row["Score Effect"])
        }
action_names = list(actions_config.keys())

# --- UI: Template Customization ---
st.subheader("🛠️ Customize Student Templates")
st.markdown("Edit the probabilities for each template using your custom actions. **Probabilities will automatically normalize to 100% when running.**")

# Dynamically build the default template dataframe based on action names
default_templates_df = pd.DataFrame({
    "Template Name": ["Grinder", "Slacker", "Balanced", "Hustler"],
    action_names[0]: [0.7, 0.1, 0.3, 0.2],  # e.g., Study
    action_names[1]: [0.2, 0.5, 0.3, 0.2],  # e.g., Rest
    action_names[2]: [0.1, 0.4, 0.2, 0.1],  # e.g., Party
    action_names[3]: [0.0, 0.0, 0.2, 0.5]   # e.g., Part-Time Job
})

edited_templates_df = st.data_editor(default_templates_df, num_rows="dynamic", use_container_width=True)

if st.button("🚀 Run Evolutionary Simulation", type="primary"):
    # Ensure we have exactly 4 actions
    if len(action_names) != 4:
        st.error("Please ensure exactly 4 unique action names are defined.")
    else:
        with st.spinner("Simulating generations..."):
            # Process the edited dataframe into a config dictionary
            templates_config = {}
            for _, row in edited_templates_df.iterrows():
                name = str(row["Template Name"]).strip()
                if name: 
                    # Extract raw probabilities using dynamic action names
                    raw_probs = np.array([float(row[act]) for act in action_names])
                    raw_probs = np.nan_to_num(raw_probs, 0) # Handle empty cells
                    
                    if sum(raw_probs) > 0:
                        normalized_probs = raw_probs / sum(raw_probs)
                    else:
                        normalized_probs = [0.25, 0.25, 0.25, 0.25] # Fallback
                    
                    # Map back to action names
                    templates_config[name] = {action_names[i]: normalized_probs[i] for i in range(4)}
                    
            # Run Simulation
            df_history = run_simulation(num_students, num_semesters, days_per_semester, templates_config, actions_config, fitness_weights)
            
        st.subheader("📈 Template Evolution Over Time")
        
        df_melted = df_history.melt(id_vars=["Semester"], var_name="Template", value_name="Population")
        
        fig = px.area(
            df_melted, 
            x="Semester", 
            y="Population", 
            color="Template",
            title="Student Population Strategy Distribution",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 View Raw Data & Final States"):
            st.dataframe(df_history.set_index("Semester"), use_container_width=True)