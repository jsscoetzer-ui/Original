# Originals — Telegram media deduplicator (Android MVP)

Originals signs into **your own Telegram account** using TDLib, scans chat history, extracts photos/videos, optionally compares them with Android MediaStore, then groups exact and visually-equivalent copies.

## What v0.1 does

- Telegram client login (phone/code/2FA/email flow supported by TDLib states)
- Main + archived chats
- “Only media sent by me” filter (default ON)
- Photos, videos, video notes, and image/video documents
- Downloads Telegram media locally through TDLib
- Reads Android photo/video library through MediaStore
- SHA-256 exact duplicate detection
- dual 64-bit dHash + average-hash matching for resized/recompressed images
- Three-frame video fingerprints (10% / 50% / 90%) plus duration/aspect constraints
- Picks a keeper by resolution, file size, then local-phone tie-break
- Never deletes anything
- Can export Telegram-only keepers to `Pictures/Originals` and `Movies/Originals`

## Telegram credentials

Telegram requires third-party clients to use an **API ID** and **API hash**. Create yours at:

https://my.telegram.org

Do not commit your credentials into source code. This app stores them in private app preferences on the phone.

## Build requirements

- Android Studio supporting AGP 9.4
- Android SDK 37
- JDK 17+
- Gradle 9.6

Open the project in Android Studio and Sync. If Android Studio asks for the Gradle distribution, use Gradle 9.6. The source bundle includes the wrapper properties but not a generated wrapper JAR.

The app uses:

```kotlin
implementation("io.github.tdlibx:td-android:1.8.56-RC10")
```

This is a precompiled Android TDLib AAR from Maven Central. If you prefer a different TDLib distribution, replace that dependency; application code uses the standard `org.drinkless.tdlib.Client` / `TdApi` Java API.

## Privacy model

- Telegram networking and local Telegram cache are handled by TDLib.
- Media matching is local on-device.
- No analytics SDK and no custom server are included.
- Secret Chats are intentionally disabled in the TDLib parameters for this MVP.
- Nothing is deleted automatically.

## Accuracy notes

Exact duplicates are deterministic (SHA-256).

Visual matching is conservative:
- image dHash Hamming distance <= 6
- image average-hash Hamming distance <= 5
- aspect-ratio deviation <= 3.5%
- video: 3 sampled frame hashes, average Hamming distance <= 7, max <= 12
- video duration difference <= 1.8 seconds

These thresholds are intentionally conservative and require two independent image fingerprints to reduce the chance of merging ordinary burst photos that merely look similar. They should be tuned on a real library before a destructive delete feature is added.

## Production hardening still recommended

Before Play Store/public release:

1. Replace in-memory scan report with Room/Room3 persistence so huge scans resume after process death.
2. Add foreground WorkManager jobs for multi-hour scans.
3. Add thumbnail review UI and side-by-side comparison.
4. Add a user-approved trash/delete flow using MediaStore delete requests.
5. Add cancellation, battery/charging constraints, and storage-space checks.
6. Validate the TDLib dependency/version you want to ship and pin checksums through dependency verification.
7. Run tests on Android 13–16, especially partial photo/video access on Android 14+.

## Safety principle

The app deliberately builds a clean library rather than deleting source media. A future delete feature should always show the keeper and every proposed deletion and require explicit confirmation.
