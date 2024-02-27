import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
import pandas as pd



# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="Exploratory Analysis with LIDA",
    page_icon="📊",
)
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
    
    # make a form to submit the key
    
    openai_key = st.text_input("Enter OpenAI API key:")    
    
    selected_model = st.selectbox(
            'Choose a model',
            options=models,
            index=0
            )
    temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0)
    
        
    use_cache = st.checkbox("Use cache", value=True)
    selected_dataset_label = st.selectbox(
            'Choose a dataset',
            options=[dataset["label"] for dataset in datasets],
            index=1
        )
    num_goals = st.slider(
                "Number of goals to generate",
                min_value=1,
                max_value=10,
                value=4)
    
    
    user_goal = st.sidebar.text_input("Describe Your Goal")
        


    selected_method_label = st.selectbox(
            'Choose a method',
            options=[method["label"] for method in summarization_methods],
            index=0
        )


    selected_method = summarization_methods[[
            method["label"] for method in summarization_methods].index(selected_method_label)]["label"]

        # add description of selected method in very small font to sidebar
    selected_summary_method_description = summarization_methods[[
            method["label"] for method in summarization_methods].index(selected_method_label)]["description"]

    if selected_method:
        st.markdown(
            f"<span> {selected_summary_method_description} </span>",
            unsafe_allow_html=True)

    num_visualizations = st.sidebar.slider(
                "Number of visualizations to generate",
                min_value=1,
                max_value=3,
                value=2)
        
    
    #visualization_libraries = ["seaborn", "matplotlib"]

    selected_library = "seaborn"
    #selected_library = st.sidebar.selectbox(
    #            'Choose a visualization library',
    #            options=visualization_libraries,
    #            index=0
    #        )


    
display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
selected_dataset = datasets[[dataset["label"]
                                     for dataset in datasets].index(selected_dataset_label)]["url"]
st.write("GPT mode", selected_model)
if openai_key:
    st.write("OpenAI API key:", display_key)
st.write("Datset:", selected_dataset_label)
st.write("Temperature:", temperature)
st.write("Number of goals:", num_goals)
if user_goal:
    st.write("User Goal:", user_goal)
st.write("Use cache:", use_cache)
st.write("Summarization method:", selected_method_label)
st.write("Number of visualizations:", num_visualizations)
st.write("Visualization library:", selected_library)


lida = Manager(text_gen=llm("openai", api_key=openai_key))
textgen_config = TextGenerationConfig(
        n=1,
        temperature=temperature,
        model=selected_model,
        use_cache=use_cache)
    
st.write("## Summary")
    # **** lida.summarize *****
summary = lida.summarize(
        selected_dataset,
        summary_method=selected_method,
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



st.write("## Goals")
#if st.session_state['form_one_complete']:
    
selected_goal = st.selectbox('Choose a generated goal', options=goal_questions, index=0)
selected_goal_index = goal_questions.index(selected_goal)
    
st.write(goals[selected_goal_index])
        
selected_goal_object = goals[selected_goal_index]

# visualize goal
if selected_goal_object:
    st.write("## Visualizations")
    st.write(goal_questions.index(selected_goal))    

    textgen_config = TextGenerationConfig(
                n=num_visualizations, temperature=temperature,
                model=selected_model,
                use_cache=use_cache)

            # **** lida.visualize *****
    visualizations = lida.visualize(
                summary=summary,
                goal=selected_goal_object,
                textgen_config=textgen_config,
                library=selected_library) 
    
    st.write("total visualizations", len(visualizations))   
    viz_titles = [f'Visualization {i+1}' for i in range(len(visualizations))]

    selected_viz_title = st.selectbox('Choose a visualization', options=viz_titles, index=0)
    
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
     
