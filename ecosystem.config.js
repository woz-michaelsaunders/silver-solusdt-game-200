module.exports = {
  apps : [{
    name: "solusd-rl-tp-21-game",
    script: "./main-sql.py",
    interpreter: './venv/bin/python',
    env: {
      PYTHONUNBUFFERED: "1",
    },
  }
 ],
};

