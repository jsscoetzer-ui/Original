package za.co.originals.app

import android.content.Context
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class ScanCoordinator(
    context: Context,
    private val telegram: TelegramClientManager
) {
    private val appContext = context.applicationContext
    private val telegramScanner = TelegramScanner(telegram)
    private val phoneScanner = PhoneMediaScanner(appContext)
    private val duplicateEngine = DuplicateEngine(appContext)
    private val exporter = OriginalsExporter(appContext)
    private val checkpoint = ScanCheckpointStore(appContext)
    private val reportStore = ScanReportStore(appContext)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var scanJob: Job? = null

    private val _work = MutableStateFlow<WorkState>(WorkState.Idle)
    val work: StateFlow<WorkState> = _work.asStateFlow()

    private val _report = MutableStateFlow<ScanReport?>(reportStore.load())
    val report: StateFlow<ScanReport?> = _report.asStateFlow()

    fun isRunning(): Boolean = scanJob?.isActive == true

    fun start(options: ScanOptions) {
        if (isRunning()) return
        if (telegram.authStage.value !is AuthStage.Ready) {
            _work.value = WorkState.Failed("Telegram is not ready")
            return
        }

        scanJob = scope.launch {
            try {
                checkpoint.begin(options)
                if (checkpoint.hasCheckpoint()) {
                    _work.value = WorkState.ScanningTelegram("Resuming saved checkpoint…", 0, 0)
                }

                val telegramAssets = telegramScanner.scan(options, checkpoint) { chat, done, total ->
                    _work.value = WorkState.ScanningTelegram(chat, done, total)
                }
                val phoneAssets = if (options.includePhoneLibrary) {
                    _work.value = WorkState.ScanningPhone("Reading your device media library")
                    phoneScanner.scan(options.includeImages, options.includeVideos)
                } else emptyList()

                val all = telegramAssets + phoneAssets
                val fingerprinted = duplicateEngine.fingerprint(all) { done, total ->
                    _work.value = WorkState.Fingerprinting(done, total)
                }
                val tgIds = telegramAssets.map { it.id }.toSet()
                val phoneIds = phoneAssets.map { it.id }.toSet()
                val (groups, keepers) = duplicateEngine.group(fingerprinted)
                val completedReport = ScanReport(
                    telegramAssets = fingerprinted.filter { it.id in tgIds },
                    phoneAssets = fingerprinted.filter { it.id in phoneIds },
                    duplicateGroups = groups,
                    uniqueKeepers = keepers
                )
                // Persist the completed result before deleting the resumable checkpoint.
                reportStore.save(completedReport)
                _report.value = completedReport
                checkpoint.clear()
                _work.value = WorkState.Complete("Scan complete · results saved")
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (t: Throwable) {
                _work.value = WorkState.Failed(t.message ?: t.javaClass.simpleName)
            }
        }
    }

    fun pause(message: String = "Scan paused. Start again to resume from the saved checkpoint.") {
        scanJob?.cancel()
        scanJob = null
        _work.value = WorkState.Complete(message)
    }

    fun externalFailure(message: String) {
        scanJob?.cancel()
        scanJob = null
        _work.value = WorkState.Failed(message)
    }

    fun exportTelegramOnlyKeepers() {
        val current = _report.value ?: return
        scope.launch {
            try {
                val toExport = current.uniqueKeepers.filter { it.source == MediaSource.TELEGRAM }
                val count = exporter.exportTelegramOnly(toExport) { done, total ->
                    _work.value = WorkState.Exporting(done, total)
                }
                _work.value = WorkState.Complete("Exported $count Telegram-only originals")
            } catch (t: Throwable) {
                _work.value = WorkState.Failed(t.message ?: "Export failed")
            }
        }
    }

    fun clearReport() {
        reportStore.clear()
        _report.value = null
        _work.value = WorkState.Idle
    }
}
