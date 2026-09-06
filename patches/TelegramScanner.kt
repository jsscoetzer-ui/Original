package za.co.originals.app

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.drinkless.tdlib.TdApi
import java.io.File

class TelegramScanner(private val td: TelegramClientManager) {
    suspend fun scan(
        options: ScanOptions,
        onProgress: (String, Int, Int) -> Unit
    ): List<MediaAsset> = withContext(Dispatchers.IO) {
        onProgress("Loading Telegram chats…", 0, 0)
        val chatIds = loadAllChatIds(options.includeArchivedChats) { batch ->
            onProgress("Loading Telegram chats… batch $batch", 0, 0)
        }
        val found = mutableListOf<MediaAsset>()
        if (chatIds.isEmpty()) {
            onProgress("No Telegram chats found", 0, 0)
            return@withContext found
        }

        chatIds.forEachIndexed { index, chatId ->
            val chat = runCatching { td.send(TdApi.GetChat(chatId)) }.getOrNull()
            val title = chat?.title ?: "Telegram chat"
            fun progress(detail: String) {
                onProgress("$title · $detail", index + 1, chatIds.size)
            }
            progress("${found.size} media found · opening chat")
            scanChat(chatId, options, found, ::progress)
        }
        found
    }

    private suspend fun loadAllChatIds(includeArchive: Boolean, onBatch: (Int) -> Unit): List<Long> {
        loadList(null, onBatch)
        val ids = linkedSetOf<Long>()
        runCatching { td.send(TdApi.GetChats(null, 10_000)) }.getOrNull()?.chatIds?.forEach { ids.add(it) }
        if (includeArchive) {
            val archive = TdApi.ChatListArchive()
            loadList(archive, onBatch)
            runCatching { td.send(TdApi.GetChats(archive, 10_000)) }.getOrNull()?.chatIds?.forEach { ids.add(it) }
        }
        return ids.toList()
    }

    private suspend fun loadList(list: TdApi.ChatList?, onBatch: (Int) -> Unit) {
        repeat(100) { batch ->
            onBatch(batch + 1)
            val result = runCatching { td.send(TdApi.LoadChats(list, 100)) }
            if (result.isFailure) return
        }
    }

    private suspend fun scanChat(
        chatId: Long,
        options: ScanOptions,
        out: MutableList<MediaAsset>,
        onDetail: (String) -> Unit
    ) {
        var fromMessageId = 0L
        var previousLast = Long.MIN_VALUE
        var messagesChecked = 0
        while (true) {
            val page = runCatching {
                td.send(TdApi.GetChatHistory(chatId, fromMessageId, 0, 100, false))
            }.getOrElse { return }
            if (page.messages.isEmpty()) return

            messagesChecked += page.messages.size
            onDetail("${out.size} media found · $messagesChecked messages checked")

            for (message in page.messages) {
                if (options.onlySentByMe && !message.isOutgoing) continue
                val descriptor = mediaDescriptor(message, options) ?: continue
                val shortName = descriptor.name.ifBlank { "Telegram media" }.take(42)

                val readyFile = descriptor.file.takeIf {
                    it.local.isDownloadingCompleted &&
                        it.local.path.isNotBlank() &&
                        File(it.local.path).exists()
                }

                val downloaded = readyFile ?: run {
                    onDetail("${out.size} media found · downloading $shortName")
                    val result = try {
                        withTimeoutOrNull(45_000L) {
                            td.send(TdApi.DownloadFile(descriptor.file.id, 32, 0, 0, true))
                        }
                    } catch (_: Throwable) {
                        null
                    }

                    if (result == null) {
                        try {
                            withTimeoutOrNull(3_000L) {
                                td.send(TdApi.CancelDownloadFile(descriptor.file.id, false))
                            }
                        } catch (_: Throwable) {
                            // Best effort only. One slow file must not stop the scan.
                        }
                        onDetail("${out.size} media found · skipped slow/unavailable file")
                    }
                    result
                } ?: continue

                val path = downloaded.local.path
                if (!downloaded.local.isDownloadingCompleted || path.isBlank() || !File(path).exists()) {
                    onDetail("${out.size} media found · skipped unavailable file")
                    continue
                }

                out += MediaAsset(
                    id = "tg:${message.chatId}:${message.id}:${descriptor.file.id}",
                    source = MediaSource.TELEGRAM,
                    kind = descriptor.kind,
                    displayName = descriptor.name.ifBlank { File(path).name },
                    mimeType = descriptor.mimeType,
                    sizeBytes = File(path).length().takeIf { it > 0 } ?: downloaded.size,
                    width = descriptor.width,
                    height = descriptor.height,
                    durationMs = descriptor.durationMs,
                    timestampSeconds = message.date.toLong(),
                    localPath = path,
                    telegramChatId = message.chatId,
                    telegramMessageId = message.id
                )
                onDetail("${out.size} media found · $messagesChecked messages checked")
            }

            val last = page.messages.last().id
            if (last == previousLast || last == 0L) return
            previousLast = last
            fromMessageId = last
        }
    }

    private data class Descriptor(
        val kind: MediaKind,
        val file: TdApi.File,
        val name: String,
        val mimeType: String?,
        val width: Int,
        val height: Int,
        val durationMs: Long
    )

    private fun mediaDescriptor(message: TdApi.Message, options: ScanOptions): Descriptor? = when (val c = message.content) {
        is TdApi.MessagePhoto -> if (options.includeImages) {
            val best = c.photo.sizes.maxByOrNull { it.width.toLong() * it.height.toLong() } ?: return null
            Descriptor(
                MediaKind.IMAGE,
                best.photo,
                "telegram_${message.id}.jpg",
                "image/jpeg",
                best.width,
                best.height,
                0
            )
        } else null

        is TdApi.MessageVideo -> if (options.includeVideos) {
            Descriptor(
                MediaKind.VIDEO,
                c.video.video,
                c.video.fileName,
                c.video.mimeType,
                c.video.width,
                c.video.height,
                c.video.duration * 1000L
            )
        } else null

        is TdApi.MessageVideoNote -> if (options.includeVideos) {
            Descriptor(
                MediaKind.VIDEO,
                c.videoNote.video,
                "video_note_${message.id}.mp4",
                "video/mp4",
                c.videoNote.length,
                c.videoNote.length,
                c.videoNote.duration * 1000L
            )
        } else null

        is TdApi.MessageDocument -> {
            val mime = c.document.mimeType
            when {
                options.includeImages && mime.startsWith("image/") -> Descriptor(
                    MediaKind.IMAGE,
                    c.document.document,
                    c.document.fileName,
                    mime,
                    0,
                    0,
                    0
                )
                options.includeVideos && mime.startsWith("video/") -> Descriptor(
                    MediaKind.VIDEO,
                    c.document.document,
                    c.document.fileName,
                    mime,
                    0,
                    0,
                    0
                )
                else -> null
            }
        }

        else -> null
    }
}
