# IMDC 2026 — how to submit the validation phase

The forecast files are built and locally validated. What remains needs **your account
actions** (registering a model is web-UI only). Follow these steps in order.

> **Deadline reality:** the validation deadline was **1 July 2026** and it is now 2 July.
> The organizers' 30 June email explicitly says to contact them about any submission-process
> issues — so **Step 0 is to email them.** They invited it; a ~1-day-late technical submission
> is a normal ask.

---

## Step 0 — Email the organizers first (today)

Send to **mosqlimate@gmail.com** (Fabiana Ganem / IMDC Organizing Committee). Draft:

> Subject: IMDC 2026 validation submission — Team Neural Earth (brief delay)
>
> Dear Fabiana and the IMDC Organizing Committee,
>
> Thank you for the reminder. On behalf of Team Neural Earth, our models and forecasts for
> the validation phase are complete and our public repository is documented and up to date.
> We are finalizing the platform upload and expect to complete it within the next day, just
> past the 1 July deadline. Could you please confirm this is acceptable, and let us know
> whether a model must be registered per track (dengue/chikungunya × state/city) or whether a
> single registered model suffices for our repository?
>
> Team: Neural Earth — Abdullah Al Helal (Oklahoma State University), Md Kamruzzaman Kamrul
> (Purdue University), Eashraque Jahan Easha (University of Denver).
> Repository: https://github.com/kamrul28890/3rd_imdc_purdue_neuralearth
> Platform account: kamrul28890
>
> Thank you very much for your help.
> Best regards,
> Md Kamruzzaman Kamrul (team leader), on behalf of Team Neural Earth

This both flags the slight delay and resolves the one open question (how many models to register).

---

## Step 1 — Make the repository submission-compliant

- The repo is already **public**, documented (`README.md`, `MODEL_CARD.md` with all required
  sections), tested, and reproducible.
- **Recommended:** the rules ask for the name `3rd_imdc_{institution}_{team}` (lowercase,
  institution acronym). Consider renaming on GitHub (Settings → Rename) to e.g.
  `3rd_imdc_{yourinstitution}_{yourteam}`, then update your local remote:
  `git remote set-url origin https://github.com/kamrul28890/3rd_imdc_..._....git`.
  (Not strictly required for upload — the platform accepts any public repo — but it matches
  the convention other teams follow.)
- **Ensure the working tree is committed** before uploading: the upload records the current
  git commit hash, and our upload tool refuses to run on a dirty tree so the hash truly
  matches the predictions.

## Step 2 — Register your model(s) at mosqlimate.org (web UI — only you can do this)

Log in at **https://mosqlimate.org** with your account and register a model pointing at your
repo. Metadata to enter:

| Field | Value |
|---|---|
| Repository | `kamrul28890/<your-repo>` |
| Name | e.g. "IMDC2026 dengue ensemble" |
| Description | one line (see `MODEL_CARD.md`) |
| Language | Python |
| Disease | Dengue (and a separate one for Chikungunya, if registering per disease) |
| ADM level | 1 (state) — and 2 (municipality) for the optional city tracks |
| Time resolution | Week |
| IMDC year | 2026 |

If Step 0's reply says one model per track is required, register up to **4**: dengue-state,
chikungunya-state, dengue-city, chikungunya-city. Note the exact `owner/repo` string you
register — the upload uses it verbatim.

## Step 3 — Upload the forecasts

With the `py310` env active and your `.env` present (it is — gitignored), from the repo root:

```bash
# dry-run first (validates against your registered model, uploads nothing):
KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.submission.upload <owner/your-repo>

# then publish:
KMP_DUPLICATE_LIB_OK=TRUE python -m imdc.submission.upload <owner/your-repo> --publish
```

This uploads the dengue state-level forecasts for all four seasons. (The module currently
targets the mandatory dengue-state track; the city/chikungunya files under
`submissions/validation/` are ready to upload the same way once those models are registered —
tell me and I'll extend the uploader to those tracks in one step.)

## Step 4 — Verify

- On mosqlimate.org, confirm your predictions appear under your model with the right seasons
  and 26 states.
- Confirm the repo link on the platform resolves and the commit hash matches.
- Reply to the organizers confirming completion.

---

## What is ready right now

- **231+ validated forecast files** in `submissions/validation/` across four tracks
  (dengue/chikungunya × state/city), all four seasons including the full-season 2025–26 forecast.
- A **documented, reproducible, public repo** (`MODEL_CARD.md`, `README.md`, `make reproduce`,
  64 tests, provenance in `RESULTS.md`).
- A **safe upload tool** (`imdc/submission/upload.py`) that validates before publishing and
  ties the upload to a clean git commit.

The only things I cannot do for you: **register the model** (web UI) and **click send** on the
email / upload. Everything else is done.
