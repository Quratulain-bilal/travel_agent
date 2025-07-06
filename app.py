import streamlit as st
import pandas as pd
import pycountry
import folium
from streamlit_folium import folium_static
from dotenv import load_dotenv
import os 
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig # type: ignore

# Load environment variables
load_dotenv()

# Initialize Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")
    st.stop()

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Define our agents
translator_agent = Agent(
    name="Translator Agent",
    instructions="You are a multilingual translation expert. Accurately translate text between any languages while preserving meaning, context, and cultural nuances.",
)

language_agent = Agent(
    name="Language Identification Agent",
    instructions="You are an expert at identifying languages and providing information about language usage in different regions.",
)

# SDK Framework Components
class LanguageMapSDK:
    def __init__(self):
        self.country_data = self._load_country_data()
        
    def _load_country_data(self):
        """Load country data with languages and coordinates"""
        countries = []
        for country in pycountry.countries:
            try:
                lang = pycountry.languages.get(alpha_2=country.languages[0]).name if hasattr(country, 'languages') else "Unknown"
                countries.append({
                    'name': country.name,
                    'official_name': getattr(country, 'official_name', country.name),
                    'latitude': getattr(country, 'latitude', 0),
                    'longitude': getattr(country, 'longitude', 0),
                    'language': lang
                })
            except:
                continue
        return pd.DataFrame(countries)
    
    def get_country_info(self, country_name):
        """Get country information including language"""
        country = self.country_data[self.country_data['name'].str.contains(country_name, case=False)]
        if not country.empty:
            return country.iloc[0].to_dict()
        return None
    
    def create_country_map(self, country_name):
        """Create a map highlighting the specified country"""
        country_info = self.get_country_info(country_name)
        if country_info:
            m = folium.Map(location=[country_info['latitude'], country_info['longitude']], zoom_start=6)
            folium.Marker(
                [country_info['latitude'], country_info['longitude']],
                popup=f"{country_info['name']} - Language: {country_info['language']}",
                icon=folium.Icon(color='red')
            ).add_to(m)
            return m
        return None

# Initialize SDK
sdk = LanguageMapSDK()

# Streamlit App
st.title("🌍 World Language Translator with Map (Gemini)")

# Sidebar for settings
st.sidebar.header("Settings")
if not gemini_api_key:
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

# Main App
col1, col2 = st.columns(2)

with col1:
    st.subheader("Find a Location")
    country_input = st.text_input("Enter a country or city (e.g., 'Madina'):")
    
    if country_input:
        country_info = sdk.get_country_info(country_input)
        if country_info:
            st.success(f"Found: {country_info['name']}")
            
            # Ask the language agent about this country
            lang_response = Runner.run_sync(
                language_agent,
                input=f"What language is primarily spoken in {country_info['name']}? Also provide 1-2 interesting facts about language use there.",
                run_config=config
            )
            
            st.write(lang_response.final_output)
            
            # Display map
            st.subheader(f"Map of {country_info['name']}")
            country_map = sdk.create_country_map(country_input)
            if country_map:
                folium_static(country_map)
        else:
            st.warning("Location not found. Try a different name.")

with col2:
    st.subheader("Language Translation")
    if country_input and country_info:
        target_language = st.selectbox(
            "Translate to:",
            ["Arabic", "English", "Urdu", "Spanish", "French", "German", "Chinese"],
            index=0 if country_info['language'] == "Arabic" else 1
        )
        
        text_to_translate = st.text_area("Enter text to translate:")
        
        if st.button("Translate"):
            if text_to_translate:
                with st.spinner("Translating..."):
                    translation = Runner.run_sync(
                        translator_agent,
                        input=f"Translate this to {target_language} while preserving cultural context:\n\n{text_to_translate}",
                        run_config=config
                    )
                    st.success("Translation:")
                    st.write(translation.final_output)
            else:
                st.warning("Please enter text to translate")

# Direct agent interaction
st.subheader("Ask the Language Expert")
user_question = st.text_input("Ask anything about languages or translations:")

if user_question:
    with st.spinner("Consulting the language expert..."):
        response = Runner.run_sync(
            language_agent if "language" in user_question.lower() else translator_agent,
            input=user_question,
            run_config=config
        )
        st.info(response.final_output)

# Instructions
st.markdown("""
### How to use this app:
1. Enter a country or city name in the search box
2. View the map and language information
3. Enter text to translate to/from the local language
4. Or ask questions to the language expert
""")