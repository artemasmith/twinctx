/**
 * ## Contract: Локальный индекс QMD
 *
 * ### Inputs
 * build получает каталог поколения со снимком Markdown и картой files из manifest.json.
 * serve получает корень поколений и принимает проект и вопрос на loopback.
 *
 * ### Outputs
 * build создаёт index.sqlite после проверки снимка и эмбеддингов.
 * GET /vsearch возвращает до пяти путей и хешей выбранного проекта.
 * GET /health проверяет пригодность указателя поколения без эмбеддингов.
 *
 * ### State
 * Готовый снимок читается через current. Сервер держит один открытый индекс.
 *
 * ### Multi-call invariants
 * Запрос использует одно поколение и одну коллекцию. Один векторный поиск
 * в процессе. При отмене слот освобождается после завершения работы QMD.
 *
 * ### Edge cases
 * Неверный ввод — 400. Пустая коллекция — 200 с пустым списком. Занятость —
 * 429. Неготовый индекс или сбой — 503. Истечение дедлайна — 504.
 *
 * ### Dependencies
 * @tobilu/qmd 2.9.0, SQLite, локальный EmbedGemma на CPU.
 */
import { createHash, randomUUID } from 'node:crypto';
import { lstat, readFile, readdir, realpath } from 'node:fs/promises';
import { createServer } from 'node:http';
import { basename, isAbsolute, join, relative, resolve } from 'node:path';

const model = 'hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf';
const projectSlug = /^[a-z0-9][a-z0-9._-]{0,79}$/;
const sha256 = /^[0-9a-f]{64}$/;
const port = 26358;
const limit = 5;
const deadlineMs = 6000;
// The SDK drops the model after five minutes without inference.
const warmIntervalMs = 3 * 60 * 1000;

process.env.QMD_EMBED_MODEL = model;

const { createStore } = await import('@tobilu/qmd');

function record(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function markdownPath(path) {
  return typeof path === 'string'
    && /\.md$/i.test(path)
    && !/[\\\x00-\x1f\x7f]/.test(path)
    && path.split('/').every((segment) => segment && !segment.startsWith('.'));
}

function collectionName(project) {
  // SDK collection names do not accept the full project slug alphabet.
  return 'p-' + Buffer.from(project, 'ascii').toString('hex');
}

async function readManifest(directory) {
  const manifest = JSON.parse(await readFile(join(directory, 'manifest.json'), 'utf8'));
  if (!record(manifest) || !record(manifest.files)) throw Error('invalid manifest');

  for (const [project, files] of Object.entries(manifest.files)) {
    if (!projectSlug.test(project) || !record(files)) throw Error('invalid project');
    for (const [path, hash] of Object.entries(files)) {
      if (!markdownPath(path) || typeof hash !== 'string' || !sha256.test(hash)) {
        throw Error('invalid source in manifest');
      }
    }
  }
  if (manifest.memory !== undefined) {
    if (!record(manifest.memory)) throw Error('invalid memory manifest');
    for (const [path, item] of Object.entries(manifest.memory)) {
      if (!record(item) || typeof item.id !== 'string' || !item.id
        || !sha256.test(item.sha256) || path !== createHash('sha256').update(item.id).digest('hex') + '.md') {
        throw Error('invalid memory identity');
      }
    }
    manifest.files['@memory'] = Object.fromEntries(
      Object.entries(manifest.memory).map(([path, item]) => [path, item.sha256])
    );
  }
  return { files: manifest.files, memory: manifest.memory };
}

async function inspectSnapshot(directory, files) {
  const collections = [];
  for (const [project, sources] of Object.entries(files)) {
    if (!Object.keys(sources).length) continue;

    const root = join(directory, 'docs', project);
    if (await realpath(root) !== root) throw Error('noncanonical snapshot directory');
    const items = await readdir(root, { recursive: true, withFileTypes: true });
    const documents = items.filter((item) => !item.isDirectory());
    if (documents.length !== Object.keys(sources).length) throw Error('snapshot count mismatch');

    for (const item of documents) {
      const path = join(item.parentPath, item.name);
      const name = relative(root, path);
      if (!item.isFile() || !Object.hasOwn(sources, name)) throw Error('unexpected snapshot file');
      const hash = createHash('sha256').update(await readFile(path)).digest('hex');
      if (hash !== sources[name]) throw Error('snapshot checksum mismatch');
    }
    collections.push([collectionName(project), { path: root, pattern: '**/*' }]);
  }
  return Object.fromEntries(collections);
}

async function build(path) {
  const directory = await realpath(path);
  const dbPath = join(directory, 'index.sqlite');
  const { files } = await readManifest(directory);
  const collections = await inspectSnapshot(directory, files);
  // An empty snapshot is a valid generation after the last document is removed.
  const store = await createStore({ dbPath, config: { collections, models: { embed: model } } });
  try {
    const expected = Object.values(files).reduce((count, sources) => count + Object.keys(sources).length, 0);
    if (expected > 0) {
      await store.update();
      const embedding = await store.embed({ maxDocsPerBatch: 8 });
      if (embedding.errors) throw Error('incomplete embedding');
    }
    const status = await store.getStatus();
    if (status.totalDocuments !== expected || status.needsEmbedding || (expected > 0 && !status.hasVectorIndex)) {
      throw Error('incomplete index');
    }
  } finally {
    await store.close();
  }
}

function reply(response, status, value) {
  if (response.writableEnded || response.destroyed) return;
  if (value === undefined) {
    response.writeHead(status).end();
    return;
  }
  response.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  }).end(JSON.stringify(value));
}

async function serve(path) {
  const root = await realpath(path);
  let current;
  let ready;
  let lastInference = 0;
  let busy = false;

  async function readyDirectory() {
    const pointer = join(root, 'current');
    if (!(await lstat(pointer)).isSymbolicLink()) throw Error('invalid generation pointer');
    const directory = await realpath(pointer);
    const target = relative(root, directory);
    if (!target || target === '..' || target.startsWith('../') || isAbsolute(target)) {
      throw Error('generation outside state directory');
    }
    const index = await lstat(join(directory, 'index.sqlite'));
    if (!index.isFile() || index.size === 0) throw Error('index not ready');
    return directory;
  }

  async function load() {
    const directory = await readyDirectory();
    if (current?.directory === directory) return current;

    const dbPath = join(directory, 'index.sqlite');
    const { files, memory } = await readManifest(directory);
    const previous = current;
    current = undefined;
    if (previous) await previous.store.close();

    const store = await createStore({ dbPath });
    current = { directory, files, memory, store };
    return current;
  }

  async function prepareGeneration() {
    if (busy) return;
    busy = true;
    try {
      const directory = await readyDirectory();
      if (ready === directory && Date.now() - lastInference < warmIntervalMs) return;
      ready = undefined;
      const { store, files } = await load();
      const first = Object.keys(files).find((project) => Object.keys(files[project]).length > 0);
      if (first) {
        // A fixed probe can hit QMD's persistent embedding cache after restart.
        const probe = await store.searchVector('контрольный прогрев ' + randomUUID(), {
          collection: collectionName(first), limit: 1, signal: AbortSignal.timeout(100_000),
        });
        if (!probe.length) throw Error('invalid warmup result');
      }
      if (await readyDirectory() === directory) {
        ready = directory;
        lastInference = Date.now();
      }
    } catch {
      ready = undefined;
    } finally {
      busy = false;
    }
  }

  async function query(request, response) {
    const url = new URL(request.url, 'http://127.0.0.1');
    if (request.method === 'GET' && url.pathname === '/health') {
      try {
        const directory = await readyDirectory();
        if (ready !== directory) throw Error("generation warming");
        reply(response, 200, { generation: basename(directory) });
      } catch {
        reply(response, 503);
      }
      return;
    }
    const memoryQuery = url.pathname === '/memory';
    const project = memoryQuery ? '@memory' : url.searchParams.get('project');
    const question = url.searchParams.get('q')?.trim();
    if (request.method !== 'GET' || (!memoryQuery && url.pathname !== '/vsearch')
      || !project || (!memoryQuery && !projectSlug.test(project)) || !question || question.length > 500) {
      reply(response, 400);
      return;
    }
    if (busy) {
      reply(response, 429);
      return;
    }
    busy = true;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      reply(response, 504);
      controller.abort();
    }, deadlineMs);
    const cancel = () => controller.abort();
    response.once('close', cancel);
    try {
      const directory = await readyDirectory();
      if (ready !== directory || current?.directory !== directory) {
        reply(response, 503);
        return;
      }
      const { files, store } = current;
      if (memoryQuery && !current.memory) throw Error("memory index not configured");
      const hits = [];
      if (Object.hasOwn(files, project) && Object.keys(files[project]).length > 0) {
        const collection = collectionName(project);
        const prefix = 'qmd://' + collection + '/';
        const results = await store.searchVector(question, { collection, limit, signal: controller.signal });
        if (!results.length) throw Error("nonempty collection returned no vectors");
        lastInference = Date.now();
        for (const result of results) {
          if (result.collectionName !== collection || !result.filepath.startsWith(prefix)) {
            throw Error('result outside project collection');
          }
          const source = result.filepath.slice(prefix.length);
          if (!Object.hasOwn(files[project], source) || files[project][source] !== result.hash) {
            throw Error('result differs from manifest');
          }
          if (hits.length < limit && !hits.some((hit) => hit.path === source)) {
            const hit = { path: source, sha256: files[project][source] };
            if (memoryQuery) {
              hit.id = current.memory[source].id;
              hit.score = result.score;
            }
            hits.push(hit);
          }
        }
      }
      reply(response, 200, { hits, generation: basename(directory) });
    } catch {
      reply(response, 503);
    } finally {
      response.off('close', cancel);
      clearTimeout(timer);
      busy = false;
    }
  }

  const server = createServer({ requestTimeout: deadlineMs }, (request, response) => {
    query(request, response).catch(() => reply(response, 400));
  });
  server.listen(port, '127.0.0.1');
  void prepareGeneration();
  setInterval(prepareGeneration, 5000).unref();
}

const [, , command, path, ...extra] = process.argv;
try {
  if (extra.length || !path || !['build', 'serve'].includes(command)) {
    throw Error('expected build or serve with one state directory');
  }
  await (command === 'build' ? build : serve)(resolve(path));
} catch {
  process.stderr.write('QMD operation failed.\n');
  process.exitCode = 1;
}
