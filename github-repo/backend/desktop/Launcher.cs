using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

[assembly: AssemblyTitle("三维重力场正演工作台")]
[assembly: AssemblyProduct("Gravity Lab")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

namespace GravityDesktop
{
    internal static class Program
    {
        [STAThread]
        private static int Main(string[] args)
        {
            if (args.Length == 3 && args[0] == "--test-session")
                return TestSession(Path.GetFullPath(args[1]), Path.GetFullPath(args[2]));
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            using (var form = new Launcher())
                Application.Run(form);
            return 0;
        }

        private static int TestSession(string ready, string stop)
        {
            try
            {
                using (var process = new LocalServer())
                {
                    var cache = Path.Combine(Path.GetDirectoryName(ready), "runtime");
                    var url = process.Start(cache);
                    File.WriteAllText(ready, url, new UTF8Encoding(false));
                    var deadline = DateTime.UtcNow.AddMinutes(10);
                    while (!File.Exists(stop) && !process.Exited && DateTime.UtcNow < deadline)
                        Thread.Sleep(150);
                }
                return 0;
            }
            catch (Exception error)
            {
                File.WriteAllText(ready + ".error", error.ToString(), new UTF8Encoding(false));
                return 1;
            }
        }
    }

    internal sealed class LocalServer : IDisposable
    {
        private static readonly string Prefix = "Gravity Lab | ";
        private readonly object gate = new object();
        private readonly StringBuilder errors = new StringBuilder();
        private Process child;
        private bool disposed;

        internal bool Exited { get { lock (gate) return child != null && child.HasExited; } }

        internal string Start(string cache)
        {
            var directory = Unpack(cache);
            var app = Path.Combine(directory, "app");
            var info = new ProcessStartInfo
            {
                FileName = Path.Combine(directory, "python.exe"),
                Arguments = "-I -B -u backend\\server.py --port 0 --device cpu --thesis " + Quote(app),
                WorkingDirectory = app,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };
            info.EnvironmentVariables["PATH"] = directory + ";" + Environment.SystemDirectory + ";" + Environment.GetFolderPath(Environment.SpecialFolder.Windows);
            var ready = new TaskCompletionSource<string>();
            lock (gate)
            {
                if (disposed) throw new ObjectDisposedException("LocalServer");
                child = new Process { StartInfo = info, EnableRaisingEvents = true };
                child.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e)
                {
                    if (e.Data == null || !e.Data.StartsWith(Prefix, StringComparison.Ordinal)) return;
                    Uri url;
                    if (Uri.TryCreate(e.Data.Substring(Prefix.Length), UriKind.Absolute, out url) &&
                        url.Scheme == "http" && url.Host == "127.0.0.1" && url.Port > 0)
                        ready.TrySetResult(url.AbsoluteUri);
                };
                child.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e)
                {
                    if (e.Data == null) return;
                    lock (errors) if (errors.Length < 10000) errors.AppendLine(e.Data);
                };
                child.Exited += delegate { ready.TrySetResult(null); };
                child.Start();
                child.BeginErrorReadLine();
                child.BeginOutputReadLine();
            }
            if (!ready.Task.Wait(TimeSpan.FromSeconds(75)) || ready.Task.Result == null)
            {
                string details;
                lock (errors) details = errors.ToString().Trim();
                throw new InvalidOperationException(details.Length == 0 ? "本地计算服务未能启动。" : details);
            }
            return ready.Task.Result;
        }

        private static string Quote(string value)
        {
            return "\"" + value + "\"";
        }

        private static string Unpack(string cache)
        {
            using (var self = File.OpenRead(Assembly.GetExecutingAssembly().Location))
            using (var archive = new ZipArchive(self, ZipArchiveMode.Read))
            using (var idReader = new StreamReader(archive.GetEntry("bundle-id.txt").Open()))
            {
                var id = Guid.ParseExact(idReader.ReadToEnd().Trim(), "N").ToString("N");
                var destination = Path.Combine(cache, id);
                using (var mutex = new Mutex(false, @"Local\GravityLab-" + id))
                {
                    bool locked;
                    try { locked = mutex.WaitOne(TimeSpan.FromMinutes(4)); }
                    catch (AbandonedMutexException) { locked = true; }
                    if (!locked) throw new TimeoutException("运行环境准备超时。");
                    try
                    {
                        if (File.Exists(Path.Combine(destination, "complete"))) return destination;
                        Directory.CreateDirectory(cache);
                        var staging = Path.Combine(cache, id + "-" + Guid.NewGuid().ToString("N"));
                        Directory.CreateDirectory(staging);
                        var root = Path.GetFullPath(staging) + Path.DirectorySeparatorChar;
                        try
                        {
                            foreach (var entry in archive.Entries)
                            {
                                if (entry.FullName == "bundle-id.txt" || entry.Name.Length == 0) continue;
                                var filename = Path.GetFullPath(Path.Combine(root, entry.FullName));
                                if (!filename.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                                    throw new InvalidDataException("Invalid package path.");
                                Directory.CreateDirectory(Path.GetDirectoryName(filename));
                                using (var input = entry.Open())
                                using (var output = new FileStream(filename, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                                    input.CopyTo(output);
                            }
                            File.WriteAllText(Path.Combine(staging, "complete"), id);
                            Directory.Move(staging, destination);
                        }
                        finally
                        {
                            if (Directory.Exists(root)) Directory.Delete(root, true);
                        }
                    }
                    finally { mutex.ReleaseMutex(); }
                }
                return destination;
            }
        }

        public void Dispose()
        {
            lock (gate)
            {
                if (disposed) return;
                disposed = true;
                if (child == null) return;
                try
                {
                    if (!child.HasExited)
                    {
                        child.Kill();
                        child.WaitForExit(4000);
                    }
                }
                catch (InvalidOperationException) { }
                finally { child.Dispose(); }
            }
        }
    }

    internal sealed class Launcher : Form
    {
        private readonly LocalServer server = new LocalServer();
        private readonly Label state = new Label();
        private readonly Button open = new Button();
        private string url;

        internal Launcher()
        {
            Text = "三维重力场正演工作台";
            Icon = Icon.ExtractAssociatedIcon(Assembly.GetExecutingAssembly().Location);
            StartPosition = FormStartPosition.CenterScreen;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            AutoScaleMode = AutoScaleMode.Dpi;
            ClientSize = new Size(400, 145);
            Padding = new Padding(20);
            Font = new Font("Microsoft YaHei UI", 10);
            state.AutoEllipsis = true;
            state.Text = "正在准备本地运行环境，首次启动需要解包。";
            state.SetBounds(20, 22, 360, 45);
            open.Text = "打开工作台";
            open.Enabled = false;
            open.SetBounds(20, 88, 126, 35);
            open.Click += delegate { OpenBrowser(); };
            var quit = new Button { Text = "退出" };
            quit.SetBounds(267, 88, 113, 35);
            quit.Click += delegate { Close(); };
            Controls.AddRange(new Control[] { state, open, quit });
            Shown += BeginLaunch;
            FormClosed += delegate { server.Dispose(); };
        }

        private async void BeginLaunch(object sender, EventArgs args)
        {
            try
            {
                var cache = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "GravityLab", "runtime");
                url = await Task.Run(() => server.Start(cache));
                if (IsDisposed) return;
                state.Text = "本地服务运行中：" + url;
                open.Enabled = true;
                OpenBrowser();
            }
            catch (Exception error)
            {
                if (IsDisposed) return;
                state.Text = "启动失败";
                MessageBox.Show(this, error.Message, Text, MessageBoxButtons.OK, MessageBoxIcon.Error);
                Close();
            }
        }

        private void OpenBrowser()
        {
            try { Process.Start(new ProcessStartInfo(url) { UseShellExecute = true }); }
            catch (Exception)
            {
                MessageBox.Show(this, url, "请在浏览器中打开", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }
    }
}
