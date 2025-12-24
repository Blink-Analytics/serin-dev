# Groq API TypeScript Test

## How to run this test

### 1. Prerequisites
- Node.js v18 or newer (Node.js v20+ recommended)
- npm (comes with Node.js)
- A valid Groq API key in your `.env` file

### 2. Setup Steps

```bash
# Navigate to the typescript test folder
cd d:\Serin\serin-dev\imp-difficulty-ranker\v0.1.1\typescript

# Initialize npm (if not already done)
npm init -y

# Install required dependencies
npm install groq-sdk dotenv xlsx
```

### 3. .env File

Make sure you have a `.env` file at:
```
d:\Serin\serin-dev\imp-difficulty-ranker\.env
```
with the following content:
```
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 4. Run the Test

You can run the test in either of the following ways:

#### Option A: Compile TypeScript and run with Node.js
```bash
npx tsc groq_test.ts
node groq_test.js
```

#### Option B: Run TypeScript directly (if you have ts-node installed)
```bash
npx ts-node groq_test.ts
```

### 5. Output

You should see the Groq API response printed in your terminal.

---

### Troubleshooting

- **No output or errors?**
  - Double-check your `.env` file path and contents.
  - Ensure your API key is correct.
  - Make sure there are no stray `export {}` or `Exp` lines in your script.
- **Module errors?**
  - Ensure all dependencies are installed (`groq-sdk`, `dotenv`).
- **Still stuck?**
  - Delete any compiled files and try again.
  - Try running the script as plain JavaScript (`node groq_test.js`).

---