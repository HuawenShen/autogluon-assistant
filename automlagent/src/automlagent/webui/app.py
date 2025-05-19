import os
from copy import deepcopy

import streamlit as st
import streamlit.components.v1 as components

from automlagent.webui.pages.start_page import main as start_page




def main():
    st.set_page_config(
    page_title="AutoGluon Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    )
    start_page()


if __name__ == "__main__":
    main()