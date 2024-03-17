import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
import pandas as pd

def init_session_state():
    if 'second_submit' not in st.session_state:
        st.session_state.second_submit = False
   
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None

    
    
    
def form_callback():
    #st.write("form callback")
    
    #st.write("selected dataset label: ", st.session_state.selected_dataset)
    st.write("selected dataset: ", st.session_state.selected_dataset)
    st.write("selected model: ", st.session_state.selected_model)
    st.write("Num goals:" , st.session_state.num_goals)
    st.write("Temperature: ", st.session_state.temperature)
    
   
    #st.write("Selected goal: ", st.session_state.selected_item)
    #st.write("User goal: ", st.session_state.user_goal)


init_session_state()

selected_library = "seaborn"
# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="Exploratory Analysis with LIDA",
    page_icon="📊",
)

openai_key= st.secrets["openai_key"]
selected_dataset = None
user_goal = None
submitted = None
models = ["gpt-3.5-turbo-16k"]
datasets = [
        
        {"label": "Cars", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"},
        {"label": "Titanic", "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"},
        {"label": "Penguins", "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"},
        ]



summarization_methods = [
        {"label": "llm",
         "description":
         "Uses the LLM to generate annotate the default summary, adding details such as semantic types for columns and dataset description"},
        {"label": "default",
         "description": "Uses dataset column statistics and column names as the summary"},

        {"label": "columns", "description": "Uses the dataset column names as the summary"}

     ]

st.write("## Exploratory Data Analysis with LIDA 📊  :bulb:")
lida=None


with st.sidebar:
    with st.form(key="input_form"):
        selected_dataset_label = st.selectbox(
            'Choose a dataset',
            options=[dataset["label"] for dataset in datasets],
            index=1,
            key="selected_dataset_label"
        )
        selected_dataset = datasets[[dataset["label"] for dataset in datasets].index(selected_dataset_label)]["url"]

        if not selected_dataset:
            st.info("Please select a dataset")
            st.stop()

        st.session_state.selected_dataset = selected_dataset

        selected_model= st.selectbox(
            'Choose a model',
            options=models,
            index=0,
            key="selected_model"
        )

        temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0, 
                key="temperature")
        
        use_cache = st.checkbox("Use Cache", key="use_cache")
        num_goals = st.slider("Number of goals", min_value=1, max_value=5, value=3, step=1, key="num_goals")
        user_goal = st.text_input("Add a goal", key="user_goal")

        selected_method_label = st.selectbox(
            'Choose a summarization method',
            options=[method["label"] for method in summarization_methods],
            index=0,
            key="selected_method_label"
        )

        selected_method = summarization_methods[[method["label"] for method in summarization_methods].index(st.session_state.selected_method_label)]["label"]
        st.session_state.selected_method = selected_method

        selected_summary_method_description = summarization_methods[[
            method["label"] for method in summarization_methods].index(selected_method_label)]["description"]


        if selected_method:
            st.markdown(
                f"<span> {selected_summary_method_description} </span>",
                unsafe_allow_html=True)
            
        num_visualizations = 1

        submitted= st.form_submit_button("Submit")
        if submitted:
            st.session_state.second_submit = True

            
    
if st.session_state.second_submit == True:

    lida = Manager(text_gen=llm("openai", api_key=openai_key))
    textgen_config = TextGenerationConfig(
        n=1,
        temperature=st.session_state.temperature,
        model=st.session_state.selected_model,
        use_cache=st.session_state.use_cache)
    
    st.write("## Summary")
    


    # **** lida.summarize *****
    summary = lida.summarize(
        st.session_state.selected_dataset,
        summary_method=st.session_state.selected_method,
        textgen_config=textgen_config)

    if "dataset_description" in summary:
        st.write(summary["dataset_description"])

    if "fields" in summary:
        fields = summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["column"] = field["column"]
            # flatted_fields["dtype"] = field["dtype"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
            # flatted_fields = {**flatted_fields, **field["properties"]}
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)
        st.write(nfields_df)
    else:
        st.write(str(summary))

    #generate goals

    goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)
    default_goal = goals[0].question
    goal_questions = [goal.question for goal in goals]


    #append user goal
    if user_goal:
        new_goal = Goal(index=0,question=user_goal, visualization=user_goal, rationale="")
        goals.append(new_goal)
        goal_questions.append(new_goal.question)
    #st.write(goal_questions)



    st.write("## Goals")
    
    selected_goal = st.selectbox('Choose a generated goal', options=goal_questions, index=0)
    selected_goal_index = goal_questions.index(selected_goal)
    
    #st.write(goals[selected_goal_index])
        
    selected_goal_object = goals[selected_goal_index]
    st.session_state.selected_item = selected_goal_object
    
    
    st.write(st.session_state.selected_item)
    
    # visualize goal
    if  st.session_state.selected_item:
        st.write("## Visualization")
        

        textgen_config = TextGenerationConfig(
                n=num_visualizations, temperature=temperature,
                model=selected_model,
                use_cache=use_cache)

            # **** lida.visualize *****
        visualizations = lida.visualize(
                summary=summary,
                #goal=selected_goal_object,
                goal=st.session_state.selected_item,
                textgen_config=textgen_config,
                library=selected_library) 
    
        #st.write("total visualizations", len(visualizations))   
        viz_titles = [f'Visualization {i+1}' for i in range(len(visualizations))]

        selected_viz_title = st.selectbox('Choose a visualization', options=viz_titles, index=0)
        #selected_viz_title = 'Visualization 1'
        selected_viz = visualizations[viz_titles.index(selected_viz_title)]

        if selected_viz.raster:
            from PIL import Image
            import io
            import base64

            imgdata = base64.b64decode(selected_viz.raster)
            img = Image.open(io.BytesIO(imgdata))
            st.image(img, caption=selected_viz_title, use_column_width=True)

            st.write("### Visualization Code")
            st.code(selected_viz.code)
     
     
