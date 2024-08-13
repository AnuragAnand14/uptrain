import streamlit as st
from openai import OpenAI
from uptrain import EvalLLM, QueryResolution, ConversationGuidelineAdherence, ConversationSatisfaction, ConversationNumberOfTurns, JailbreakDetection, CustomPromptEval, Settings
import pandas as pd
import json

def evaluate_jailbreak(prompt, openai_api_key):
    eval_llm = EvalLLM(openai_api_key=openai_api_key)
    data = [{"question": prompt}]
    result = eval_llm.evaluate(data=data, checks=[JailbreakDetection()])[0]
    score = result.get("score_jailbreak_attempted", None)
    
    if score == 1.0:
        return "Yes"
    elif score == 0.0:
        return "No"
    else:
        return "NA"

class MetricsEvaluator:
    def __init__(self, openai_api_key):
        self.eval_llm = EvalLLM(openai_api_key=openai_api_key)

    def evaluate_metrics(self, data):
        urgency_prompt = """
        You are an expert in evaluating the urgency of query and then forward them to human help if they are very urgent.
        Salutation and Interjections like Hello, Hi, How are You, Thank You should always be in Others category and given a 0.0 score.
        You are evaluating the following query:
        {question}
        Please determine the urgency of this query and provide your response as one of the following:
        - Very Urgent and has to be forwarded to Human Help
        - Very Urgent but can be resolved by LLM
        - Urgent
        - Normal
        - Not at all Urgent
        - Others
        """
        choices = ["Very Urgent and has to be forwarded to Human Help ", "Very Urgent but can be resolved by LLM", "Urgent", "Normal", "Not at all Urgent"]
        choice_scores = [100, 80, 60, 40, 20, 00]
        prompt_var_to_column_mapping = {"question": "user_question"}

        results = self.eval_llm.evaluate(
            data=data,
            checks=[
                QueryResolution(user_persona="User", llm_persona=" Advisor Chatbot"),
                ConversationGuidelineAdherence(
                    guideline="Provide relevant advice related to Internet issues only from the user queries without being too detailed",
                    guideline_name="Guidance"
                ),
                ConversationSatisfaction(user_persona="User", llm_persona=" Advisor Chatbot"),
                ConversationNumberOfTurns(user_persona="User", llm_persona=" Advisor Chatbot"),
                CustomPromptEval(
                    prompt=urgency_prompt,
                    choices=choices,
                    choice_scores=choice_scores,
                    prompt_var_to_column_mapping=prompt_var_to_column_mapping
                )
            ]
        )
        return results

@st.cache_data(show_spinner=False)
def get_evaluation_scores(chat_data, openai_api_key):
    metrics_evaluator = MetricsEvaluator(openai_api_key=openai_api_key)
    results = metrics_evaluator.evaluate_metrics(chat_data)
    if isinstance(results, list) and len(results) > 0:
        result = results[0]
        return {
            "Query Resolution": result.get('score_query_resolution', 0.0) * 100,
            "Guideline Adherence": result.get('score_conversation_Guidance_adherence', 0.0) * 100,
            "Conversation Satisfaction": result.get('score_conversation_satisfaction', 0.0) * 100,
            "Conversation Length": result.get('conversation_length', 0.0) ,
            "Urgency": result.get('score_custom_prompt', 0.0) ,
        }
   
    return None


def display_score_bars(scores_list):
    if not scores_list:
        st.write("No scores available yet.")
        return

    for scores in scores_list:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            qr_score = scores.get('Query Resolution', 0.0)
            st.progress(qr_score / 100, text=f"QueryRes: {qr_score:.2f}")
        with col2:
            ga_score = scores.get('Guideline Adherence', 0.0)
            st.progress(ga_score / 100, text=f"GuideAdherence: {ga_score:.2f}")
        with col3:
            cs_score = scores.get('Conversation Satisfaction', 0.0)
            st.progress(cs_score / 100, text=f"Satisfaction: {cs_score:.2f}")
        with col4:
            urgency_score = scores.get('Urgency')
            if urgency_score is not None:
                st.progress(urgency_score / 100, text=f"Urgency: {urgency_score:.2f}")
            else:
                st.write("Urgency: N/A")

    

st.set_page_config(layout="wide")

st.write("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.user-message {
    align-self: flex-start;
    background-color: #e0ffe0;
    padding: 10px;
    border-radius: 10px;
    max-width: 80%;
    float: left;
    clear: both;
}
.assistant-message {
    align-self: flex-end;
    background-color: #f0f0f5;
    padding: 10px;
    border-radius: 10px;
    max-width: 80%;
    float: right;
    clear: both;
}
.assistant-avatar {
    margin-right: 10px;
}
.message-content {
    display: inline-block;
    vertical-align: top;
    max-width: 80%;
}
.stProgress > div > div > div {
    background-color: #4caf50;
}
.score-bar {
    display: flex;
    flex-direction: row;
    gap: 10px;
    margin-top: 10px;
}
.score-bar .stProgress {
    flex: 1;
}
</style>
""", unsafe_allow_html=True)

if "show_scores" not in st.session_state:
    st.session_state.show_scores = True 

if "messages" not in st.session_state:
    st.session_state.messages = []
if "cumulative_scores" not in st.session_state:
    st.session_state.cumulative_scores = None
if "human_feedback" not in st.session_state:
    st.session_state.human_feedback = None
if "individual_scores" not in st.session_state:
    st.session_state.individual_scores = []

if st.button("Reset Chat", key="reset_top_left"):
    st.session_state.messages = []
    st.session_state.cumulative_scores = None
    st.session_state.human_feedback = None
    st.session_state.individual_scores = []
    st.rerun()

chat_col, score_col = st.columns([3, 1])

with chat_col:
    st.title("💬 Helpful Chatbot")
    st.write(
        "This is an  chatbot that uses OpenAI's GPT-3.5 model. "
        "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
    )
    

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    else:
        client = OpenAI(api_key=openai_api_key)

    chat_container = st.container()
    input_container = st.container()

    with input_container:
        with input_container:
               prompt = st.chat_input("Your Query Here")
        

    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f'''
                    <div class="user-message">
                        <div class="user-avatar">👤</div>
                        <div class="message-content">{message["content"]}</div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div class="assistant-message">
                        <div class="assistant-avatar">🤖</div>
                        <div class="message-content">{message["content"]}</div>
                    </div>
                ''', unsafe_allow_html=True)
            if i > 0 and i % 2 == 1 and i == len(st.session_state.messages) and st.session_state.show_scores and st.session_state.individual_scores:
                display_score_bars([st.session_state.individual_scores[-1]])

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown(f'''
                <div class="user-message">
                    <div class="user-avatar">👤</div>
                    <div class="message-content">{prompt}</div>
                </div>
            ''', unsafe_allow_html=True)

            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            messages.append({"role": "system", "content": "Act as user internet service executive that that helps with relevant advice to the the users query,do not give a very detailed answer."})

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            ).choices[0].message.content

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(f'''
                <div class="assistant-message">
                    <div class="assistant-avatar">🤖</div>
                    <div class="message-content">{response}</div>
                </div>
            ''', unsafe_allow_html=True)

            # Evaluate the last pair (up to 8 messages)
            last_pair = [{"conversation": st.session_state.messages[-2:], "user_question": prompt}]
            current_scores = get_evaluation_scores(last_pair, openai_api_key)
            jailbreak_score = evaluate_jailbreak(prompt, openai_api_key)
            current_scores["Jailbreak Attempt"] = jailbreak_score

            # Evaluate the whole chat
            whole_chat = [{"conversation": st.session_state.messages, "user_question": st.session_state.messages[0]["content"]}]
            cumulative_scores = get_evaluation_scores(whole_chat, openai_api_key)
            cumulative_scores["Jailbreak Attempt"] = "Yes" if jailbreak_score == "Yes" or (st.session_state.cumulative_scores and st.session_state.cumulative_scores["Jailbreak Attempt"] == "Yes") else "No"
            cumulative_scores["Conversation Length"] = len(st.session_state.messages) 

            # Store scores
            st.session_state.individual_scores.append({
                "Pair": len(st.session_state.individual_scores) + 1,
                "User": prompt,
                "Assistant": response,
                **current_scores
            })
            st.session_state.cumulative_scores = cumulative_scores

            # Display individual score bars for the current pair
            display_score_bars([current_scores])

with score_col:
    st.write("### Cumulative Evaluation Scores")
    if st.session_state.cumulative_scores:
        # Create a DataFrame from the cumulative scores
        df_cumulative = pd.DataFrame([st.session_state.cumulative_scores])
        
        # Display the table
        st.table(df_cumulative)

    else:
        st.write("No scores available yet.")

    # Display log of all pair scores
    st.write("### Pairwise Dialogue Scores")
    if st.session_state.individual_scores:
        # Create a DataFrame from all individual scores
        df_log = pd.DataFrame(st.session_state.individual_scores)
        
        # Display the table
        st.dataframe(df_log)
    else:
        st.write("No pair scores available yet.")

# Print the evaluation results
print(json.dumps(st.session_state.individual_scores, indent=2))
print(json.dumps(st.session_state.cumulative_scores, indent=2))
