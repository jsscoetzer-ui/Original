package za.co.originals.app

import android.content.Context
import android.net.Uri
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

class ScanReportStore(context: Context) {
    private val file = File(context.filesDir, "completed_scan_report.json")
    private val lock = Any()

    fun save(report: ScanReport) = synchronized(lock) {
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(encodeReport(report).toString())
        if (file.exists() && !file.delete()) error("Could not replace previous scan report")
        if (!tmp.renameTo(file)) {
            tmp.copyTo(file, overwrite = true)
            tmp.delete()
        }
    }

    fun load(): ScanReport? = synchronized(lock) {
        if (!file.exists()) return@synchronized null
        runCatching { decodeReport(JSONObject(file.readText())) }.getOrNull()
    }

    fun clear() = synchronized(lock) {
        if (file.exists()) file.delete()
    }

    private fun encodeReport(report: ScanReport): JSONObject {
        val all = LinkedHashMap<String, MediaAsset>()
        (report.telegramAssets + report.phoneAssets + report.uniqueKeepers + report.duplicateGroups.flatMap { it.members })
            .forEach { all[it.id] = it }

        return JSONObject().apply {
            put("version", 1)
            put("assets", JSONArray().apply { all.values.forEach { put(encodeAsset(it)) } })
            put("telegramIds", JSONArray(report.telegramAssets.map { it.id }))
            put("phoneIds", JSONArray(report.phoneAssets.map { it.id }))
            put("keeperIds", JSONArray(report.uniqueKeepers.map { it.id }))
            put("groups", JSONArray().apply {
                report.duplicateGroups.forEach { group ->
                    put(JSONObject().apply {
                        put("id", group.id)
                        put("type", group.type.name)
                        put("memberIds", JSONArray(group.members.map { it.id }))
                        put("keeperId", group.keeper.id)
                        put("similarity", group.similarity)
                    })
                }
            })
        }
    }

    private fun decodeReport(root: JSONObject): ScanReport {
        val assets = mutableMapOf<String, MediaAsset>()
        root.getJSONArray("assets").forEachObject { obj ->
            val asset = decodeAsset(obj)
            assets[asset.id] = asset
        }

        fun ids(name: String): List<String> = root.optJSONArray(name)?.stringList().orEmpty()
        fun select(name: String): List<MediaAsset> = ids(name).mapNotNull(assets::get)

        val groups = mutableListOf<DuplicateGroup>()
        root.optJSONArray("groups")?.forEachObject { obj ->
            val members = obj.getJSONArray("memberIds").stringList().mapNotNull(assets::get)
            val keeper = assets[obj.getString("keeperId")] ?: return@forEachObject
            groups += DuplicateGroup(
                id = obj.getString("id"),
                type = MatchType.valueOf(obj.getString("type")),
                members = members,
                keeper = keeper,
                similarity = obj.optDouble("similarity", 0.0)
            )
        }

        return ScanReport(
            telegramAssets = select("telegramIds"),
            phoneAssets = select("phoneIds"),
            duplicateGroups = groups,
            uniqueKeepers = select("keeperIds")
        )
    }

    private fun encodeAsset(asset: MediaAsset): JSONObject = JSONObject().apply {
        put("id", asset.id)
        put("source", asset.source.name)
        put("kind", asset.kind.name)
        put("name", asset.displayName)
        if (asset.mimeType != null) put("mime", asset.mimeType)
        put("size", asset.sizeBytes)
        put("width", asset.width)
        put("height", asset.height)
        put("duration", asset.durationMs)
        put("timestamp", asset.timestampSeconds)
        if (asset.localPath != null) put("path", asset.localPath)
        if (asset.contentUri != null) put("uri", asset.contentUri.toString())
        if (asset.telegramChatId != null) put("chat", asset.telegramChatId)
        if (asset.telegramMessageId != null) put("message", asset.telegramMessageId)
        if (asset.sha256 != null) put("sha256", asset.sha256)
        if (asset.visualHash != null) put("visualHash", asset.visualHash)
        if (asset.averageHash != null) put("averageHash", asset.averageHash)
        if (asset.videoHashes != null) put("videoHashes", JSONArray(asset.videoHashes.toList()))
    }

    private fun decodeAsset(j: JSONObject): MediaAsset = MediaAsset(
        id = j.getString("id"),
        source = MediaSource.valueOf(j.getString("source")),
        kind = MediaKind.valueOf(j.getString("kind")),
        displayName = j.optString("name", "media"),
        mimeType = j.optString("mime").takeIf { it.isNotBlank() },
        sizeBytes = j.optLong("size", 0L),
        width = j.optInt("width", 0),
        height = j.optInt("height", 0),
        durationMs = j.optLong("duration", 0L),
        timestampSeconds = j.optLong("timestamp", 0L),
        localPath = j.optString("path").takeIf { it.isNotBlank() },
        contentUri = j.optString("uri").takeIf { it.isNotBlank() }?.let(Uri::parse),
        telegramChatId = if (j.has("chat")) j.optLong("chat") else null,
        telegramMessageId = if (j.has("message")) j.optLong("message") else null,
        sha256 = j.optString("sha256").takeIf { it.isNotBlank() },
        visualHash = if (j.has("visualHash")) j.optLong("visualHash") else null,
        averageHash = if (j.has("averageHash")) j.optLong("averageHash") else null,
        videoHashes = j.optJSONArray("videoHashes")?.let { a -> LongArray(a.length()) { index -> a.optLong(index) } }
    )

    private inline fun JSONArray.forEachObject(block: (JSONObject) -> Unit) {
        for (i in 0 until length()) block(getJSONObject(i))
    }

    private fun JSONArray.stringList(): List<String> = buildList {
        for (i in 0 until length()) add(optString(i))
    }
}
