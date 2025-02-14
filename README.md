# Introduction to Artificial Intelligence - Assignments

This repository contains assignments for the **Introduction to Artificial Intelligence** course.

## Setting up the environment

To ensure a consistent development environment, we use Python's built-in `venv` module to create a virtual environment.

### Prerequisites

- **Python 3.10 or later** (Check your version by running `python --version` or `python3 --version`)
- **pip** (Should be included with Python, verify with `pip --version`)

### Installation Steps

#### 1. Clone the Repository
```sh
git clone <repository_url>
cd <repository_name>
```

#### 2. Create a Virtual Environment
```sh
python -m venv venv
```
This will create a `venv/` directory containing the virtual environment.

#### 3. Activate the Virtual Environment
- **Windows (PowerShell):**
  ```sh
  venv\Scripts\Activate
  ```
- **macOS/Linux:**
  ```sh
  source venv/bin/activate
  ```
After activation, you should see `(venv)` at the beginning of your terminal prompt.

#### 4. Install Dependencies
```sh
pip install -r requirements.txt
```
This installs all required packages listed in `requirements.txt`.

#### 5. Verify Installation
```sh
python -c "import numpy, pandas, matplotlib, gymnasium; print('Setup complete!')"
```
If no errors appear, the setup is successful.

### Deactivating the Virtual Environment
When you're done working, deactivate the virtual environment:
```sh
deactivate
```

## Running Scripts
To run any Python script, make sure the virtual environment is activated and execute:
```sh
python your_script.py
```

## Troubleshooting
- If installation fails due to missing dependencies, ensure `pip` is up to date:
  ```sh
  pip install --upgrade pip
  ```
- If using Windows and encountering execution policy errors, try:
  ```sh
  Set-ExecutionPolicy Unrestricted -Scope Process
  ```
  in PowerShell before activating `venv`.

---

Your environment should now be properly set up! 🚀
