import { useEffect, useRef, useState } from "react";
import "./App.css";

// バックエンド(FastAPI)が返す1件分の結果の型
type Result = { label: string; prob: number };

// 切り替え可能なデータセット（＝内部で使う学習済みモデル）
type Dataset = {
  id: string;
  label: string;
  mockResults: Result[];
};

const DATASETS: Dataset[] = [
  {
    id: "cifar10",
    label: "CIFAR-10（検証用）",
    mockResults: [
      { label: "airplane", prob: 0.55 },
      { label: "automobile", prob: 0.3 },
      { label: "bird", prob: 0.15 },
    ],
  },
  {
    id: "glass-beads",
    label: "古代ガラス玉 産地推定（本番想定）",
    mockResults: [
      { label: "候補産地 A", prob: 0.68 },
      { label: "候補産地 B", prob: 0.21 },
      { label: "候補産地 C", prob: 0.11 },
    ],
  },
];

// 画像キュー1件分の状態
type QueueStatus = "queued" | "processing" | "done" | "error";

type QueueItem = {
  id: string;
  file: File;
  previewUrl: string;
  status: QueueStatus;
  results: Result[];
  errorMessage: string | null;
  checked: boolean; // 保存対象として選択されているか（デフォルトtrue）
};

// 保存済み一覧1件分の記録（IndexedDBにそのまま保存する形）
type SavedRecord = {
  id: string;
  sourceId: string; // どのキュー項目から保存されたか（重複保存の判定に使う）
  fileName: string;
  imageBlob: Blob; // 画像本体。ブラウザを閉じても残るようIndexedDBに保存する
  savedAt: number; // 保存日時（epoch ms）
  results: Result[];
  note: string; // 備考（編集可能）
};

// 画面表示用：DBの記録に、表示専用のオブジェクトURLを加えたもの
type SavedItem = SavedRecord & { previewUrl: string };

// ---- IndexedDBへの永続化 ------------------------------------------
// ページを再読み込みしても「保存済み」タブのデータが消えないようにする
const DB_NAME = "glass-bead-db";
const DB_VERSION = 1;
const STORE_NAME = "savedItems";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE_NAME)) {
        req.result.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbGetAll(): Promise<SavedRecord[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).getAll();
    req.onsuccess = () => resolve(req.result as SavedRecord[]);
    req.onerror = () => reject(req.error);
  });
}

async function dbPut(record: SavedRecord): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put(record);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// 複数件を1つのトランザクションでまとめて書き込む（1件ずつputするより効率的）
async function dbPutMany(records: SavedRecord[]): Promise<void> {
  if (records.length === 0) return;
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    records.forEach((record) => store.put(record));
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbDelete(id: string): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

function toRecord(item: SavedItem): SavedRecord {
  return {
    id: item.id,
    sourceId: item.sourceId,
    fileName: item.fileName,
    imageBlob: item.imageBlob,
    savedAt: item.savedAt,
    results: item.results,
    note: item.note,
  };
}
// --------------------------------------------------------------------

// 「すべて分類する」実行時の同時処理数（実バックエンドへの負荷を抑えるため）
const CONCURRENCY = 3;

// 保存済み一覧を1ページに表示する件数（多くなっても全件DOM描画しないため）
const SAVED_PAGE_SIZE = 20;

// ---- 設定 -------------------------------------------------------
const API_URL = "http://localhost:8000/predict";
// true の間はサーバーに接続せず、ダミーの結果を表示する。
// バックエンドができたら false に変更する。
const USE_MOCK = true;
// -----------------------------------------------------------------

async function requestPrediction(file: File, datasetId: string): Promise<Result[]> {
  if (USE_MOCK) {
    await new Promise((resolve) => setTimeout(resolve, 800)); // 通信の待ち時間を再現
    const dataset = DATASETS.find((d) => d.id === datasetId) ?? DATASETS[0];
    return dataset.mockResults;
  }

  const form = new FormData();
  form.append("file", file); // サーバー側の引数名 "file" と合わせる
  form.append("dataset", datasetId); // どの学習済みモデルを使うか

  const res = await fetch(API_URL, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(`サーバーがエラーを返しました (${res.status})`);
  }
  const data = await res.json();
  return data.results as Result[];
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// 予測結果カード + Top3内訳。key に選択中の画像IDを渡すことで、表示対象が
// 切り替わるたびに新規マウントとなり、バーが0%から伸びるアニメーションが
// 毎回再生される。
function ResultDetail({ results }: { results: Result[] }) {
  const [barsReady, setBarsReady] = useState(false);

  useEffect(() => {
    const id = requestAnimationFrame(() => {
      requestAnimationFrame(() => setBarsReady(true));
    });
    return () => cancelAnimationFrame(id);
  }, []);

  const top3 = results.slice(0, 3);
  const top1 = top3[0];
  const runnerUps = top3.slice(1); // 2位・3位のみ（1位は予測結果カードと重複するため除外）

  if (!top1) return null;

  return (
    <>
      <div className="top-pick">
        <span className="top-pick-tag">予測結果</span>
        <div className="top-pick-main">
          <span className="top-pick-label">{top1.label}</span>
          <span className="top-pick-prob">{(top1.prob * 100).toFixed(1)}%</span>
        </div>
        <div className="bar large">
          <div
            className="bar-fill"
            style={{ width: barsReady ? `${top1.prob * 100}%` : "0%" }}
          />
        </div>
      </div>

      {runnerUps.length > 0 && (
        <details className="runner-ups">
          <summary className="subheading">他の候補を見る（2〜3位）</summary>
          <ol className="results" start={2}>
            {runnerUps.map((r, i) => (
              <li key={r.label}>
                <div className="row">
                  <span className="rank">{i + 2}</span>
                  <span className="label">{r.label}</span>
                  <span className="prob">{(r.prob * 100).toFixed(1)}%</span>
                </div>
                <div className="bar">
                  <div
                    className="bar-fill"
                    style={{
                      width: barsReady ? `${r.prob * 100}%` : "0%",
                      transitionDelay: `${i * 90}ms`,
                    }}
                  />
                </div>
              </li>
            ))}
          </ol>
        </details>
      )}
    </>
  );
}

// 保存済み一覧タブ：検索・並び替え・備考編集・削除
type SortKey = "fileName" | "savedAt" | "result";
type SortDir = "asc" | "desc";

function SavedTab({
  items,
  onUpdateNote,
  onDelete,
}: {
  items: SavedItem[];
  onUpdateNote: (id: string, note: string) => void;
  onDelete: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("savedAt");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftNote, setDraftNote] = useState("");
  const [page, setPage] = useState(1);

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  function startEdit(item: SavedItem) {
    setEditingId(item.id);
    setDraftNote(item.note);
  }

  function confirmEdit(id: string) {
    onUpdateNote(id, draftNote);
    setEditingId(null);
  }

  function sortIndicator(key: SortKey) {
    if (key !== sortKey) return null;
    return (
      <span className="sort-arrow" aria-hidden="true">
        {sortDir === "asc" ? "▲" : "▼"}
      </span>
    );
  }

  const q = query.trim().toLowerCase();
  const filtered = items.filter((it) => {
    if (!q) return true;
    const label = it.results[0]?.label ?? "";
    return (
      it.fileName.toLowerCase().includes(q) ||
      label.toLowerCase().includes(q) ||
      it.note.toLowerCase().includes(q)
    );
  });

  const sorted = [...filtered].sort((a, b) => {
    let cmp = 0;
    if (sortKey === "fileName") cmp = a.fileName.localeCompare(b.fileName, "ja");
    else if (sortKey === "savedAt") cmp = a.savedAt - b.savedAt;
    else cmp = (a.results[0]?.label ?? "").localeCompare(b.results[0]?.label ?? "", "ja");
    return sortDir === "asc" ? cmp : -cmp;
  });

  // 件数が多くても表を全件DOM描画しないよう、ページ単位に区切って表示する
  const totalPages = Math.max(1, Math.ceil(sorted.length / SAVED_PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pageItems = sorted.slice(
    (currentPage - 1) * SAVED_PAGE_SIZE,
    currentPage * SAVED_PAGE_SIZE,
  );

  return (
    <section className="panel saved-panel" aria-labelledby="saved-title">
      <div className="panel-header">
        <h2 id="saved-title">保存済み一覧</h2>
        {items.length > 0 && <span className="pill">{items.length}件</span>}
      </div>

      {items.length === 0 ? (
        <p className="placeholder">まだ保存された画像はありません</p>
      ) : (
        <>
          <input
            type="search"
            className="saved-search"
            placeholder="ファイル名・分類結果・備考で検索"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="保存済み一覧を検索"
          />

          {sorted.length === 0 ? (
            <p className="placeholder">検索条件に一致するデータがありません</p>
          ) : (
            <div className="table-wrap">
              <table className="saved-table">
                <thead>
                  <tr>
                    <th scope="col">プレビュー</th>
                    <th scope="col">
                      <button type="button" className="sort-button" onClick={() => toggleSort("fileName")}>
                        画像名{sortIndicator("fileName")}
                      </button>
                    </th>
                    <th scope="col">
                      <button type="button" className="sort-button" onClick={() => toggleSort("savedAt")}>
                        分類日時{sortIndicator("savedAt")}
                      </button>
                    </th>
                    <th scope="col">
                      <button type="button" className="sort-button" onClick={() => toggleSort("result")}>
                        分類結果{sortIndicator("result")}
                      </button>
                    </th>
                    <th scope="col">備考</th>
                    <th scope="col">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {pageItems.map((item) => {
                    const top1 = item.results[0];
                    const isEditing = editingId === item.id;
                    return (
                      <tr key={item.id}>
                        <td>
                          <img className="saved-thumb" src={item.previewUrl} alt="" />
                        </td>
                        <td className="saved-filename">{item.fileName}</td>
                        <td>{new Date(item.savedAt).toLocaleString("ja-JP")}</td>
                        <td>{top1 ? `${top1.label}（${(top1.prob * 100).toFixed(1)}%）` : "―"}</td>
                        <td>
                          {isEditing ? (
                            <input
                              type="text"
                              className="note-input"
                              value={draftNote}
                              onChange={(e) => setDraftNote(e.target.value)}
                              aria-label={`${item.fileName} の備考`}
                              autoFocus
                            />
                          ) : (
                            <span className="note-text">{item.note || "―"}</span>
                          )}
                        </td>
                        <td>
                          <div className="row-actions">
                            {isEditing ? (
                              <>
                                <button
                                  type="button"
                                  className="icon-button"
                                  onClick={() => confirmEdit(item.id)}
                                >
                                  保存
                                </button>
                                <button
                                  type="button"
                                  className="icon-button"
                                  onClick={() => setEditingId(null)}
                                >
                                  キャンセル
                                </button>
                              </>
                            ) : (
                              <>
                                <button
                                  type="button"
                                  className="icon-button"
                                  onClick={() => startEdit(item)}
                                >
                                  編集
                                </button>
                                <button
                                  type="button"
                                  className="icon-button danger"
                                  onClick={() => onDelete(item.id)}
                                >
                                  削除
                                </button>
                              </>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {totalPages > 1 && (
            <div className="pager">
              <button
                type="button"
                className="icon-button"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={currentPage <= 1}
              >
                ← 前へ
              </button>
              <span className="pager-status">
                {currentPage} / {totalPages} ページ（{sorted.length}件）
              </span>
              <button
                type="button"
                className="icon-button"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage >= totalPages}
              >
                次へ →
              </button>
            </div>
          )}
        </>
      )}
    </section>
  );
}

type Tab = "classify" | "saved";

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("classify");
  const [items, setItems] = useState<QueueItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [datasetId, setDatasetId] = useState(DATASETS[0].id);
  const [isDragOver, setIsDragOver] = useState(false);
  const [savedItems, setSavedItems] = useState<SavedItem[]>([]);
  // 同じ画像を二重に自動保存しないためのガード（保存処理が完了するまでの間だけ記録する）
  const savingIdsRef = useRef<Set<string>>(new Set());

  // 起動時にIndexedDBから保存済みデータを読み込む
  useEffect(() => {
    dbGetAll()
      .then((records) => {
        setSavedItems(
          records.map((r) => ({ ...r, previewUrl: URL.createObjectURL(r.imageBlob) })),
        );
      })
      .catch((err) => console.error("保存済みデータの読み込みに失敗しました", err));
  }, []);

  // 明示的に選択されたIDが無効（未選択 or 削除済み）なら先頭の画像にフォールバックする
  const selectedItem = items.find((it) => it.id === selectedId) ?? items[0] ?? null;

  const totalCount = items.length;
  const doneCount = items.filter((it) => it.status === "done").length;
  const isBatchRunning = items.some((it) => it.status === "processing");
  // エラーになった画像も「すべて分類する」の対象に含める（再試行の唯一の手段のため）
  const pendingCount = items.filter((it) => it.status === "queued" || it.status === "error").length;

  const savedSourceIds = new Set(savedItems.map((s) => s.sourceId));

  // チェックが付いた画像の分類が完了したら、ボタン操作なしで自動的にDBへ保存する
  useEffect(() => {
    const alreadySaved = new Set(savedItems.map((s) => s.sourceId));
    const toSave = items.filter(
      (it) =>
        it.checked &&
        it.status === "done" &&
        !alreadySaved.has(it.id) &&
        !savingIdsRef.current.has(it.id),
    );
    if (toSave.length === 0) return;

    toSave.forEach((it) => savingIdsRef.current.add(it.id));

    // 1件ずつ setSavedItems を呼ぶと保存件数が増えるほど配列コピーが繰り返され重くなるため、
    // レコード作成とDB書き込みはまとめて行い、state更新もバッチ内で1回だけにする
    const records: SavedRecord[] = toSave.map((it) => ({
      id: crypto.randomUUID(),
      sourceId: it.id,
      fileName: it.file.name,
      imageBlob: it.file,
      savedAt: Date.now(),
      results: it.results,
      note: "",
    }));

    (async () => {
      try {
        await dbPutMany(records);
        setSavedItems((prev) => [
          ...prev,
          ...records.map((record) => ({
            ...record,
            previewUrl: URL.createObjectURL(record.imageBlob),
          })),
        ]);
      } catch (err) {
        console.error("画像の自動保存に失敗しました", err);
      } finally {
        toSave.forEach((it) => savingIdsRef.current.delete(it.id));
      }
    })();
  }, [items, savedItems]);

  // 画像を追加（複数可）：既存のキューに追加する
  function addFiles(fileList: FileList | File[]) {
    const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
    if (files.length === 0) return;

    const newItems: QueueItem[] = files.map((file) => ({
      id: crypto.randomUUID(),
      file,
      previewUrl: URL.createObjectURL(file),
      status: "queued",
      results: [],
      errorMessage: null,
      checked: true,
    }));
    setItems((prev) => [...prev, ...newItems]);
  }

  function handleFileInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files) addFiles(e.target.files);
    e.target.value = ""; // 同じファイルを再度選び直せるようにする
  }

  function handleDragOver(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragOver(true);
  }

  function handleDragLeave(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragOver(false);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files) addFiles(e.dataTransfer.files);
  }

  function toggleChecked(id: string) {
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, checked: !it.checked } : it)));
  }

  function removeItem(id: string) {
    setItems((prev) => {
      const target = prev.find((it) => it.id === id);
      if (target) URL.revokeObjectURL(target.previewUrl);
      return prev.filter((it) => it.id !== id);
    });
    setSelectedId((prev) => (prev === id ? null : prev));
  }

  function clearAll() {
    setItems((prev) => {
      prev.forEach((it) => URL.revokeObjectURL(it.previewUrl));
      return [];
    });
    setSelectedId(null);
  }

  // データセット（使用モデル）が切り替わったとき：別モデルの結果を引きずらないよう全件リセット
  function handleDatasetChange(e: React.ChangeEvent<HTMLSelectElement>) {
    setDatasetId(e.target.value);
    setItems((prev) =>
      prev.map((it) => ({ ...it, status: "queued", results: [], errorMessage: null })),
    );
  }

  // 1件だけ分類する（「すべて分類する」から呼ばれる）
  async function classifyItem(id: string) {
    const item = items.find((it) => it.id === id);
    if (!item || item.status === "processing") return;

    setItems((prev) =>
      prev.map((it) => (it.id === id ? { ...it, status: "processing", errorMessage: null } : it)),
    );
    try {
      const results = await requestPrediction(item.file, datasetId);
      setItems((prev) => prev.map((it) => (it.id === id ? { ...it, status: "done", results } : it)));
    } catch (err) {
      const message = err instanceof Error ? err.message : "予測に失敗しました";
      setItems((prev) =>
        prev.map((it) => (it.id === id ? { ...it, status: "error", errorMessage: message, results: [] } : it)),
      );
    }
  }

  // 「すべて分類する」：待機中・エラーの画像を、同時実行数を制限しながらまとめて処理する
  async function classifyAll() {
    const queue = items
      .filter((it) => it.status === "queued" || it.status === "error")
      .map((it) => it.id);
    let cursor = 0;

    async function worker() {
      while (cursor < queue.length) {
        const id = queue[cursor];
        cursor += 1;
        await classifyItem(id);
      }
    }

    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, queue.length) }, worker));
  }

  function updateSavedNote(id: string, note: string) {
    setSavedItems((prev) => prev.map((s) => (s.id === id ? { ...s, note } : s)));
    const target = savedItems.find((s) => s.id === id);
    if (target) {
      dbPut(toRecord({ ...target, note })).catch((err) =>
        console.error("備考の保存に失敗しました", err),
      );
    }
  }

  function deleteSavedItem(id: string) {
    setSavedItems((prev) => {
      const target = prev.find((s) => s.id === id);
      if (target) URL.revokeObjectURL(target.previewUrl);
      return prev.filter((s) => s.id !== id);
    });
    dbDelete(id).catch((err) => console.error("保存データの削除に失敗しました", err));
  }

  return (
    <main className="app">
      <header className="hero">
        <p className="eyebrow">ANCIENT GLASS BEAD ANALYSIS</p>
        <h1>古代ガラス玉 産地推定システム</h1>

        {activeTab === "classify" && (
          <div className="hero-controls">
            <label htmlFor="dataset-select" className="hero-select-label">
              使用モデル（データセット）
            </label>
            <select
              id="dataset-select"
              className="hero-select"
              value={datasetId}
              onChange={handleDatasetChange}
            >
              {DATASETS.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>
        )}
      </header>

      <nav className="tabs" aria-label="表示の切り替え">
        <button
          type="button"
          className={`tab-button${activeTab === "classify" ? " is-active" : ""}`}
          onClick={() => setActiveTab("classify")}
        >
          分類
        </button>
        <button
          type="button"
          className={`tab-button${activeTab === "saved" ? " is-active" : ""}`}
          onClick={() => setActiveTab("saved")}
        >
          保存済み
          {savedItems.length > 0 && <span className="tab-count">{savedItems.length}</span>}
        </button>
      </nav>

      {activeTab === "classify" ? (
        <div className="layout">
          {/* 左：画像キュー（複数画像をまとめて管理） */}
          <section className="panel" aria-labelledby="queue-title">
            <div className="panel-header">
              <h2 id="queue-title">画像</h2>
              {totalCount > 0 && <span className="pill">{totalCount}枚選択中</span>}
            </div>

            <div
              className={`dropzone${isDragOver ? " is-dragover" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <label className="button secondary">
                画像を追加
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleFileInputChange}
                  hidden
                />
              </label>
              <p className="dropzone-hint">またはここに画像をドラッグ&ドロップ（複数選択可）</p>
            </div>

            {totalCount > 0 && (
              <>
                <div className="queue-toolbar">
                  <button
                    className="button primary"
                    onClick={classifyAll}
                    disabled={pendingCount === 0 || isBatchRunning}
                  >
                    {isBatchRunning ? (
                      <span className="button-loading">
                        <span className="spinner" aria-hidden="true" />
                        処理中…
                      </span>
                    ) : (
                      `すべて分類する（${pendingCount}枚）`
                    )}
                  </button>
                  <button className="button secondary" onClick={clearAll}>
                    クリア
                  </button>
                </div>

                <div className="queue-progress">
                  <div className="bar">
                    <div
                      className="bar-fill"
                      style={{ width: `${(doneCount / totalCount) * 100}%` }}
                    />
                  </div>
                  <span className="queue-progress-text">
                    {doneCount}/{totalCount} 処理済み
                  </span>
                </div>
              </>
            )}

            {totalCount > 0 && (
              <ul className="queue-list">
                {items.map((item) => (
                  <li key={item.id}>
                    <label className="queue-checkbox">
                      <input
                        type="checkbox"
                        checked={item.checked}
                        onChange={() => toggleChecked(item.id)}
                        aria-label={`${item.file.name} を自動保存の対象に含める`}
                      />
                    </label>
                    <button
                      type="button"
                      className={`queue-row${item.id === selectedItem?.id ? " is-selected" : ""}`}
                      onClick={() => setSelectedId(item.id)}
                      aria-current={item.id === selectedItem?.id ? "true" : undefined}
                    >
                      <img className="queue-thumb" src={item.previewUrl} alt="" />
                      <span className="queue-row-main">
                        <span className="queue-filename">{item.file.name}</span>
                        <span className={`queue-status queue-status-${item.status}`}>
                          {item.status === "queued" && "待機中"}
                          {item.status === "processing" && (
                            <>
                              <span className="spinner" aria-hidden="true" />
                              解析中…
                            </>
                          )}
                          {item.status === "done" &&
                            item.results[0] &&
                            `${item.results[0].label} ${(item.results[0].prob * 100).toFixed(1)}%`}
                          {item.status === "error" && "エラー"}
                          {savedSourceIds.has(item.id) && (
                            <span className="queue-saved-badge" title="保存済み">
                              ✓
                            </span>
                          )}
                        </span>
                      </span>
                      <span className="queue-chevron" aria-hidden="true">
                        ›
                      </span>
                    </button>
                    <button
                      type="button"
                      className="queue-remove"
                      aria-label={`${item.file.name} をキューから削除`}
                      onClick={() => removeItem(item.id)}
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          {/* 右：選択中の画像の詳細 */}
          <section className="panel" aria-labelledby="result-title">
            <div className="panel-header">
              <h2 id="result-title">選択中の画像</h2>
              {selectedItem && (
                <p className="filename">
                  <span className="filename-icon" aria-hidden="true">🖼</span>
                  <span className="filename-text">{selectedItem.file.name}</span>
                  <span className="filename-size">{formatBytes(selectedItem.file.size)}</span>
                </p>
              )}
            </div>

            {!selectedItem && (
              <p className="placeholder">画像を追加すると、ここに詳細が表示されます</p>
            )}

            {selectedItem && (
              <>
                <div className="preview">
                  <img src={selectedItem.previewUrl} alt="選択したガラス玉の画像" />
                </div>

                {selectedItem.status === "error" && (
                  <p className="error">{selectedItem.errorMessage}</p>
                )}

                {selectedItem.status === "processing" && (
                  <div className="loading-state">
                    <span className="spinner large" aria-hidden="true" />
                    <p className="placeholder">解析中です…</p>
                  </div>
                )}

                {selectedItem.status === "queued" && (
                  <p className="placeholder">
                    「すべて分類する」を押すとこの画像も解析されます
                  </p>
                )}

                {selectedItem.status === "done" && (
                  <ResultDetail key={selectedItem.id} results={selectedItem.results} />
                )}
              </>
            )}
          </section>
        </div>
      ) : (
        <SavedTab items={savedItems} onUpdateNote={updateSavedNote} onDelete={deleteSavedItem} />
      )}
    </main>
  );
}
