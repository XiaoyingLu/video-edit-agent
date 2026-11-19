# Soundtrack assistant architecture and delivery plan

## Product decisions

- Discovery and original generation are both in scope; catalog search is delivered first.
- Generation uses hosted providers initially, behind a provider-neutral adapter so local generation can be added later.
- Commercial use and monetized-video use are mandatory by default.
- A candidate is never described as "copyright-free" merely because it is free to download. It must have documented rights suitable for the intended use.

## Architecture

```text
Video project artifacts
  ├─ creative brief, story treatment, storyboard
  ├─ scene understanding and optional footage/timeline context
  └─ target duration and editing style
             |
             v
     SoundtrackBrief
       ├─ mood and genre directions
       ├─ instrumentation and vocal preference
       ├─ tempo range and narrative energy arc
       └─ search terms and generation prompt
             |
       +-----+----------------+
       |                      |
       v                      v
 Catalog search          Hosted generation
 Provider adapters       Provider adapters
       |                      |
       +----------+-----------+
                  v
          Candidate normalization
                  |
           License validation
                  |
          Candidate ranking
                  |
          Preview against video
                  |
          Explicit user selection
                  |
          Existing music analysis
                  |
          Existing music-aware edit review
                  |
             Approved render
```

### Core boundaries

- `SoundtrackBrief` is a content-derived request, not a track selection or a claim about rights.
- Catalog and generation providers implement narrow interfaces and return normalized candidates. Catalog search uses an async provider protocol and a bounded, provider-neutral search request/response contract.
- Provider-specific HTTP clients, credentials, rate limits, and response parsing stay inside provider adapters.
- License validation is independent of provider search and generation. Unknown or incomplete rights fail closed. The initial evaluator treats CC BY-NC and CC BY-ND as ineligible and CC BY-SA as requiring explicit review; provider-supplied permission claims must still be backed by license evidence.
- Ranking is explainable and separates creative fit from legal eligibility. A high creative score cannot override a failed rights check.
- The existing FFmpeg renderer and music-analysis pipeline remain provider-agnostic. This review stage records an explicit selection but does not download the asset or alter the timeline; a later integration stage will hand an authorized local asset to the existing analysis workflow.
- All selection and timeline changes remain human-approved; discovery and generation never render or alter a timeline automatically.

### Candidate rights and provenance

Every candidate must retain, where available:

- provider and provider asset ID;
- title, creator, duration, tags, and source page;
- license name/type, canonical license URL, and version or captured license text;
- explicit commercial-use and monetized-video permissions;
- attribution text and whether attribution is required;
- adaptation/remix and synchronization permission;
- download URL or generation job ID, retrieval date, and local asset checksum;
- provider terms snapshot/reference and any account/plan constraints.

Do not treat "royalty-free", "free", "publicly available", or a provider marketing label as sufficient proof. Reject non-commercial-only, no-derivatives, unclear, or unverifiable licenses for the default policy. Share-alike licenses require explicit review because downstream video distribution may trigger obligations. Attribution-required tracks may be eligible, but the required credit must be retained and surfaced in export metadata/documentation. Generated tracks require the same review: generation does not itself guarantee copyright ownership, exclusivity, or commercial-use permission.

## First catalog integration: Soundstripe

Soundstripe publishes an official REST API for listing songs with search, duration, BPM, mood/genre/instrument tags, instrumental filtering, pagination, and related artist/audio resources. Its API offering states that licensing is handled at the integration layer and tracks delivered through the API are pre-cleared. API access is account-keyed; trial keys are watermarked and expire after 30 days, while production access is subject to the applicable agreement. The adapter therefore requires `SOUNDSTRIPE_API_KEY` and accepts an explicit `SoundtrackLicense` profile. It does not infer commercial, monetized-video, or synchronization rights from the marketing phrase “royalty-free”; populate that profile only after confirming the account-specific API agreement and distribution scope.

The adapter uses the documented `/v1/songs` endpoint. It does not download audio, select tracks, or claim that an account is licensed without the supplied profile.

## Catalog contract foundation

The normalized `SoundtrackCandidate` and `SoundtrackLicense` models, async `SoundtrackCatalogProvider` protocol, bounded search request/response models, and fail-closed rights evaluator are in place. This integration-free foundation does not imply that any catalog has been approved, nor does it download tracks.

The default rights evaluator requires explicit commercial-use, monetized-video, adaptation, and synchronization permissions, along with a canonical license URL and evidence URL. Missing permission data is not treated as permission. CC BY-NC and CC BY-ND are rejected for this default workflow; CC BY-SA is held for explicit review. Attribution-required candidates need their credit text retained. These checks are conservative software gates, not legal advice or an independent legal verification of provider claims.

## Candidate search, audition, and selection

`search-soundtrack` uses the saved brief to query the Soundstripe catalog and writes `.video-agent/soundtrack-candidates.json`. `SOUNDSTRIPE_API_KEY` must be provided through the environment. An optional project-local `--license-profile` JSON file can supply rights claims only after the account agreement has been checked; without it, the provider uses an unknown license profile and candidates cannot pass the rights gate.

`review-soundtrack` ranks the normalized response and writes a JSON report plus a read-only HTML audition page. Pass `--video` with a project-local video to show it beside candidate audio players. Playback is manual; the page does not synchronize playback automatically. The report shows creative-fit score separately from rights status and includes provider details and preview links.

`select-soundtrack --candidate-id ...` records a deliberate choice only when the candidate is `eligible`. `review_required` and `ineligible` candidates are blocked. Selection is stored as `.video-agent/soundtrack-selection.json`, with provenance and rights evidence retained. This stage does not download audio, change a timeline, or render a video. Existing review artifacts are immutable; rerun search/review with a new output path if you want to preserve an earlier audition.

## Incremental PR plan

| PR | Scope | Acceptance criteria |
|---|---|---|
| #62 | Soundtrack brief from the existing video plan | Typed, deterministic brief; narrative energy arc; project-local artifact and dependency tracking; CLI command |
| #63 | Catalog search contracts and normalized candidate model | Provider-neutral async search interface; rights/provenance schema; fail-closed eligibility evaluator; no provider assumptions |
| #64 | First catalog provider | Official API/authorized download path only; documented provider terms; integration tests with mocked HTTP |
| #65 | Candidate ranking | Explainable creative-fit score; rights eligibility is a hard gate; deterministic tests |
| #66 | Preview and selection | Soundstripe search CLI; read-only HTML audition against optional project video; explainable ranking and rights status; explicit eligible-only selection artifact; preserve source files and timeline |
| #67 | Hosted generation adapter | Provider selected after terms/API review; secrets via environment; async job lifecycle; rights metadata |
| #68 | Generated-track evaluation and workflow integration | Analyze generated audio with existing analyzer; compare alternatives; human selection; no auto-render |

## Implementation rules

1. Verify current provider APIs and account-specific license terms before choosing a catalog or generation vendor.
2. Use official APIs and explicitly permitted downloads; do not scrape music sites or bypass access controls.
3. Keep provider credentials out of project artifacts and logs.
4. Persist license evidence and candidate provenance with the project.
5. Fail closed when commercial use, monetization, synchronization, or adaptation rights cannot be established.
6. Test network adapters with mocked responses; do not make external calls in unit tests.
7. Preserve the existing approval gates and deterministic rendering path.
