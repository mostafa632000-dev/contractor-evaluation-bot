import streamlit as st
import numpy as np
import pandas as pd
import re
from typing import List, Tuple



st.set_page_config(page_title="Contractor Evaluation Bot", layout="centered")

# ---------------------------- Constants and Setup ----------------------------

DEFAULT_CRITERIA = ["TP", "AT", "S", "TC", "MC", "E", "ES"]
DEFAULT_CONTRACTORS = [f"Contractor{i}" for i in range(1, 7)]

# ---------------------------- Helper Functions ----------------------------


def ensure_reciprocal(A: pd.DataFrame) -> pd.DataFrame:
    M = A.copy().astype(float)
    idx = M.index
    for i in range(len(idx)):
        M.iat[i, i] = 1.0
        for j in range(i + 1, len(idx)):
            val = M.iat[i, j]
            if val == 0 or pd.isna(val):
                val = 1.0
            M.iat[j, i] = 1.0 / val
    return M

def normalize_matrix(A: pd.DataFrame) -> pd.DataFrame:
    """ Normalize the matrix by dividing each element by the sum of its column. """
    col_sums = A.sum(axis=0)
    return A / col_sums

def calculate_weights(normalized_matrix: pd.DataFrame) -> np.ndarray:
    """ Calculate the weights by averaging the normalized values in each row. """
    w = normalized_matrix.mean(axis=1)
    return w / w.sum()  # Normalize the weights to make them sum to 1

def percentage_weights(weights: np.ndarray) -> np.ndarray:
    """ Convert the weights to percentages. """
    return weights * 100

def calculate_consistency(pairwise: pd.DataFrame, weights: np.ndarray) -> Tuple[float, float, float, float]:
    """ Calculate the consistency values: λ_max, CI, and CR. """
    AW = pairwise.values @ weights
    lambda_max = float(np.mean(AW / weights))
    n = pairwise.shape[0]
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = 1.32
    CR = CI / RI if RI != 0 else 0.0
    
    # Convert CR to percentage (multiply by 100)
    CR_percentage = CR * 100
    
    return lambda_max, CI, CR_percentage, RI

def topsis(decision: pd.DataFrame, weights: np.ndarray, benefit_mask: List[bool]) -> pd.DataFrame:
    """ Calculate the rankings using the TOPSIS method. """
    X = decision.values.astype(float)
    denom = np.sqrt((X ** 2).sum(axis=0))
    if np.any(denom == 0):
        raise ValueError("Some decision matrix columns have all zeros; provide non-zero values for each criterion.")
    R = X / denom
    V = R * weights
    V_plus = np.where(benefit_mask, V.max(axis=0), V.min(axis=0))
    V_minus = np.where(benefit_mask, V.min(axis=0), V.max(axis=0))
    S_plus = np.sqrt(((V - V_plus) ** 2).sum(axis=1))
    S_minus = np.sqrt(((V - V_minus) ** 2).sum(axis=1))
    C = S_minus / (S_plus + S_minus)
    out = pd.DataFrame({'S_plus': S_plus, 'S_minus': S_minus, 'C': C}, index=decision.index)
    out['Rank'] = out['C'].rank(ascending=False, method='min').astype(int)
    out = out.sort_values('C', ascending=False)
    return out

def display_pairwise(pairwise: pd.DataFrame):
    """ Display the pairwise comparison matrix in Streamlit. """
    st.dataframe(pairwise)

# ---------------------------- Chatbot Helper Functions ----------------------------

def bot_say(text: str):
    st.session_state.messages.append({"role": "assistant", "content": text})
    with st.chat_message("assistant"):
        st.markdown(text)

def handle_user_message(text: str):
    txt = text.strip()

    # Start the conversation
    if "start" in txt.lower():
        return """
👋 **Welcome to the Contractor Evaluation Bot!**  
We'll evaluate contractors using **AHP** to calculate criteria weights and then apply **TOPSIS** to select the best contractor.

🏗 **The 7 Evaluation Criteria:**
- ⿡ **TP** – Tender Price (overall project cost and financial competitiveness)
- ⿢ **AT** – Accomplishment Time (ability to complete work on schedule)
- ⿣ **S** – Safety (safety management and accident prevention)
- ⿤ **TC** – Technical Capability (technology, resources, and engineering quality)
- ⿥ **MC** – Management Capability (organization, supervision, coordination)
- ⿦ **E** – Experience (previous relevant project experience)
- ⿧ **ES** – Economic Status (financial stability and economic strength)

⚖ **AHP Stage**  
Please enter all **21 pairwise comparison values** for the 7 criteria 
Please enter all 21 pairwise comparison values for the 7 criteria separated by commas in the following order:

TP vs AT, TP vs S, TP vs TC, TP vs MC, TP vs E, TP vs ES

AT vs S, AT vs TC, AT vs MC, AT vs E, AT vs ES

S vs TC, S vs MC, S vs E, S vs ES

TC vs MC, TC vs E, TC vs ES

MC vs E, MC vs ES

E vs ES

"""

    # Handle user AHP input for pairwise comparison
    if "pairwise" in txt.lower():
        return """
Now, please enter the **21 pairwise comparison values** separated by commas:

Example: 3,5,2,4,6,8,2,3,1,4,5,2,3,4,5,2,3,4,2,3,2
"""
    
    # Parse the pairwise values
    if re.match(r"^(\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?)*)$", txt):  # Check for valid numeric input
        try:
            values = [float(i.strip()) for i in txt.split(",")]
            if len(values) == 21:
                # Update pairwise matrix
                A = pd.DataFrame(np.ones((7, 7)), index=DEFAULT_CRITERIA, columns=DEFAULT_CRITERIA)
                pairs = [
                    ("TP", "AT"), ("TP", "S"), ("TP", "TC"), ("TP", "MC"), ("TP", "E"), ("TP", "ES"),
                    ("AT", "S"), ("AT", "TC"), ("AT", "MC"), ("AT", "E"), ("AT", "ES"),
                    ("S", "TC"), ("S", "MC"), ("S", "E"), ("S", "ES"),
                    ("TC", "MC"), ("TC", "E"), ("TC", "ES"),
                    ("MC", "E"), ("MC", "ES"),
                    ("E", "ES")
                ]
                idx = 0
                for i, j in pairs:
                    A.loc[i, j] = values[idx]
                    A.loc[j, i] = 1 / values[idx]
                    idx += 1
                st.session_state.pairwise = A
                display_pairwise(A)

                # Normalize and calculate weights
                normalized_matrix = normalize_matrix(A)
                weights = calculate_weights(normalized_matrix)
                weights_percentage = percentage_weights(weights)

                # Calculate consistency
                lambda_max, CI, CR, RI = calculate_consistency(A, weights)

                # Display normalized matrix, weights, and consistency values
                st.session_state.weights = pd.DataFrame({"criterion": DEFAULT_CRITERIA, "weight": weights_percentage})

                result = f"""
**Computation done! Here are the results:**

### **Weights:**
- **TP**: {weights_percentage[0]:.2f}%
- **AT**: {weights_percentage[1]:.2f}%
- **S**: {weights_percentage[2]:.2f}%
- **TC**: {weights_percentage[3]:.2f}%
- **MC**: {weights_percentage[4]:.2f}%
- **E**: {weights_percentage[5]:.2f}%
- **ES**: {weights_percentage[6]:.2f}%

### **Consistency Check:**
- **λ_max**: {lambda_max:.4f}
- **Consistency Index (CI)**: {CI:.6f}
- **Random Index (RI)**: {RI:.2f}
- **Consistency Ratio (CR)**: {CR:.2f}%  

*Note: CR should be less than 10% (0.1) to ensure consistency.*

Do you want to proceed to the **TOPSIS** stage? Type **yes** to proceed or **no** to end the chat.
                """
                return result
            elif len(values) == 42:  # For TOPSIS stage (6 contractors × 7 criteria)
                # Process TOPSIS values
                decision_matrix = np.array(values).reshape(6, 7)
                df = pd.DataFrame(
                    decision_matrix,
                    columns=DEFAULT_CRITERIA,
                    index=DEFAULT_CONTRACTORS
                )
                st.session_state.decision = df
                
                # Display decision matrix
                st.dataframe(df)
                
                # Calculate TOPSIS
                weights = st.session_state.weights['weight'].values / 100  # Convert percentages back to decimals
                benefit_criteria = [True, True, True, False, True, True, True]  # TC is cost criterion
                
                try:
                    ranking = topsis(df, weights, benefit_criteria)
                    st.session_state.ranking = ranking
                    
                    # Format results
                    result = """
### **Decision Matrix has been processed!**

### **TOPSIS Rankings:**
"""
                    for idx, row in ranking.iterrows():
                        result += f"- **{idx}**: Rank {int(row['Rank'])} (Score: {row['C']:.4f})\n"
                    
                    result += "\nDo you want to start a new evaluation? Type 'start' to begin again."
                    return result
                    
                except Exception as e:
                    return f"Error in TOPSIS computation: {e}"
            else:
                return """
Please enter values in one of these formats:
- 21 comma-separated values for AHP
- 42 comma-separated values for TOPSIS (6 contractors × 7 criteria)
"""
        except Exception as e:
            return f"Error in computation: {e}"

    if "yes" in txt.lower():
        return """
Please enter 42 comma-separated values for 6 contractors (7 values each):

Example: 600,14,0.06,0.36,0.05,0.40,0.06,605,16,0.33,0.04,0.07,0.22,0.07,...

Order for each contractor: TP,AT,S,TC,MC,E,ES
"""

    if "no" in txt.lower():
        return "Thank you for using the Contractor Evaluation Bot! Goodbye."

    return "Sorry, I didn't understand. Type 'start' to begin."

# ---------------------------- Chatbot Interface ----------------------------

st.title("Contractor Evaluation Chatbot")

# Initialize the chatbot memory (if none exists)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    response = handle_user_message(user_input)
    bot_say(response)

# ---------------------------- Button for New Chat ----------------------------
if st.button("New Chat"):
    st.session_state.messages = []
    st.session_state.pairwise = pd.DataFrame(np.ones((7,7)), index=DEFAULT_CRITERIA, columns=DEFAULT_CRITERIA)
    st.session_state.weights = None
    st.session_state.ahp_stats = None
    st.session_state.decision = pd.DataFrame(index=[], columns=DEFAULT_CRITERIA, dtype=float)
    st.session_state.ranking = None
    st.session_state.clear()  # Reset session and "re-run"
