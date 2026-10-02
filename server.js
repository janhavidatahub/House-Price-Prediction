/**
 * server.js — Main Express Server for HousePriceBot Backend
 * ==========================================================
 * Exposes:
 *   GET  /health          → backend + ML service status
 *   POST /predict         → forward features to FastAPI, return prediction (used by React frontend)
 *   GET  /webhook         → Meta WhatsApp verification
 *   POST /webhook         → incoming WhatsApp messages
 *   POST /test/message    → local chatbot simulation (no WhatsApp needed)
 */

"use strict";

require("dotenv").config();

const express = require("express");
const cors    = require("cors");
const app     = express();

// ─── CORS ──────────────────────────────────────────────────────────────────
// Allow the React frontend (running on a different port, e.g. 5173) to call
// this Express server. Without this the browser blocks the request.
app.use(cors({
  origin: process.env.FRONTEND_URL || "*",
}));

app.use(express.json());

// ─── Import modules ────────────────────────────────────────────────────────
const webhookRouter                    = require("./routes/webhook");
const { processMessage }               = require("./conversation/conversationManager");
const { callMlService, checkMlHealth } = require("./services/mlService");

const PORT = process.env.PORT || 3000;

// ─── GET /health ───────────────────────────────────────────────────────────
app.get("/health", async (req, res) => {
  const mlOk = await checkMlHealth();
  res.json({
    status:               "ok",
    service:              "house-price-node-backend",
    ml_service_reachable: mlOk,
    ml_service_url:       process.env.ML_SERVICE_URL || "http://localhost:8000",
  });
});

// ─── POST /predict ─────────────────────────────────────────────────────────
// This is the primary endpoint used by the React dashboard.
// It validates the eight house features, forwards them to the Python FastAPI
// ML service, and returns the prediction JSON to the frontend.
//
// Request:  { MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude }
// Response: { predicted_price: 4.364, estimated_usd: 436395.46, note: "..." }
app.post("/predict", async (req, res) => {
  const required = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
  ];

  // Validate all eight fields are present and numeric
  for (const field of required) {
    const val = req.body[field];
    if (val === undefined || val === null || isNaN(parseFloat(val))) {
      return res.status(400).json({
        error: `Invalid or missing field: ${field}. Must be a number.`,
      });
    }
  }

  // Build a clean numeric payload (coerce strings to floats just in case)
  const payload = {};
  for (const field of required) {
    payload[field] = parseFloat(req.body[field]);
  }

  try {
    // Proxy the request to the Python FastAPI ML service
    const result = await callMlService(payload);
    return res.json(result);
  } catch (err) {
    console.error("[/predict] ML service error:", err.message);
    return res.status(503).json({
      error: "Prediction service unavailable. Make sure the Python ML service is running on port 8000.",
    });
  }
});

// ─── WhatsApp Webhook ──────────────────────────────────────────────────────
app.use("/webhook", webhookRouter);

// ─── POST /test/message ────────────────────────────────────────────────────
// Local chatbot simulation — no WhatsApp credentials needed.
app.post("/test/message", async (req, res) => {
  const { phoneNumber, message } = req.body;
  if (!phoneNumber || typeof phoneNumber !== "string") {
    return res.status(400).json({ error: 'Missing or invalid "phoneNumber".' });
  }
  if (!message || typeof message !== "string") {
    return res.status(400).json({ error: 'Missing or invalid "message".' });
  }
  try {
    const reply = await processMessage(phoneNumber, message, callMlService);
    return res.json({ reply });
  } catch (err) {
    console.error("[/test/message] Error:", err.message);
    return res.status(500).json({ error: "Internal server error", detail: err.message });
  }
});

// ─── 404 ───────────────────────────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({
    error: "Route not found",
    available_routes: [
      "GET  /health",
      "POST /predict         (React frontend)",
      "GET  /webhook         (Meta verification)",
      "POST /webhook         (WhatsApp messages)",
      "POST /test/message    (local testing)",
    ],
  });
});

// ─── Start ─────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log("╔══════════════════════════════════════════════════╗");
  console.log("║         HousePriceBot — Node.js Backend          ║");
  console.log("╚══════════════════════════════════════════════════╝");
  console.log(`  Server      : http://localhost:${PORT}`);
  console.log(`  ML service  : ${process.env.ML_SERVICE_URL || "http://localhost:8000"}`);
  console.log(`  /predict    : POST http://localhost:${PORT}/predict`);
  console.log(`  /health     : GET  http://localhost:${PORT}/health`);
  console.log("────────────────────────────────────────────────────");
  console.log("  Make sure the Python ML service is also running:");
  console.log("    cd ml-service && uvicorn app:app --reload --port 8000");
  console.log("────────────────────────────────────────────────────");
});
